from pykalman import KalmanFilter
import torch
import inspect


def array1d(X, dtype=None, order=None):
    """Returns at least 1-d array with data from X"""
    return torch.atleast_1d(X)


def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return torch.atleast_2d(X)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    # wyshi: we don't set seed here, because it will change the order of training data
    # wyshi: if we want to use em, then we need to set the seed, but in general, we should set the seed outside
    assert type(seed) is int


def _determine_dimensionality(variables, default):
    """Derive the dimensionality of the state space

    Parameters
    ----------
    variables : list of ({None, array}, conversion function, index)
        variables, functions to convert them to arrays, and indices in those
        arrays to derive dimensionality from.
    default : {None, int}
        default dimensionality to return if variables is empty

    Returns
    -------
    dim : int
        dimensionality of state space as derived from variables or default.
    """
    # gather possible values based on the variables
    candidates = []
    for (v, converter, idx) in variables:
        if v is not None:
            v = converter(v)
            candidates.append(v.shape[idx])

    # also use the manually specified default
    if default is not None:
        candidates.append(default)

    # ensure consistency of all derived values
    if len(candidates) == 0:
        return 1
    else:
        if not torch.all(torch.tensor(candidates) == candidates[0]).item():
            raise ValueError("The shape of all " + "parameters is not consistent.  " + "Please re-check their values.")
        return candidates[0]


def get_params(obj):
    """Get names and values of all parameters in `obj`'s __init__"""
    try:
        # get names of every variable in the argument
        args = inspect.getargspec(obj.__init__)[0]
        args.pop(0)  # remove "self"

        # get values for each of the above in the object
        argdict = dict([(arg, obj.__getattribute__(arg)) for arg in args])
        return argdict
    except:
        raise ValueError("object has no __init__ method")


def preprocess_arguments(argsets, converters):
    """convert and collect arguments in order of priority

    Parameters
    ----------
    argsets : [{argname: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argname: function}
        conversion functions for each argument

    Returns
    -------
    result : {argname: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for (argname, argval) in argset.items():
            # check that this argument is necessary
            if not argname in converters and argname != "device":
                raise ValueError("Unrecognized argument: {0}".format(argname))

            # potentially use this argument
            if argname not in result and argval is not None and argname in converters:
                # convert to right type
                argval = converters[argname](argval)

                # save
                result[argname] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = "The following arguments are missing: {0}".format(list(missing))
        raise ValueError(s)

    return result


def _arg_or_default(arg, default, dim, name):
    if arg is None:
        result = default
    else:
        result = arg
    if len(result.shape) > dim:
        raise ValueError(("%s is not constant for all time." + "  You must specify it manually.") % (name,))
    return result


def _filter_predict(
    transition_matrix, transition_covariance, transition_offset, current_state_mean, current_state_covariance
):
    r"""Calculate the mean and covariance of :math:`P(x_{t+1} | z_{0:t})`

    Using the mean and covariance of :math:`P(x_t | z_{0:t})`, calculate the
    mean and covariance of :math:`P(x_{t+1} | z_{0:t})`.

    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state} array
        state transition matrix from time t to t+1
    transition_covariance : [n_dim_state, n_dim_state] array
        covariance matrix for state transition from time t to t+1
    transition_offset : [n_dim_state] array
        offset for state transition from time t to t+1
    current_state_mean: [n_dim_state] array
        mean of state at time t given observations from times
        [0...t]
    current_state_covariance: [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t]

    Returns
    -------
    predicted_state_mean : [n_dim_state] array
        mean of state at time t+1 given observations from times [0...t]
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of state at time t+1 given observations from times
        [0...t]
    """
    predicted_state_mean = transition_matrix @ current_state_mean + transition_offset
    predicted_state_covariance = (
        torch.mm(transition_matrix, torch.mm(current_state_covariance, transition_matrix.T)) + transition_covariance
    )

    return (predicted_state_mean, predicted_state_covariance)


def _filter_correct(
    observation_matrix,
    observation_covariance,
    observation_offset,
    predicted_state_mean,
    predicted_state_covariance,
    observation,
):
    r"""Correct a predicted state with a Kalman Filter update

    Incorporate observation `observation` from time `t` to turn
    :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`

    Parameters
    ----------
    observation_matrix : [n_dim_obs, n_dim_state] array
        observation matrix for time t
    observation_covariance : [n_dim_obs, n_dim_obs] array
        covariance matrix for observation at time t
    observation_offset : [n_dim_obs] array
        offset for observation at time t
    predicted_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t-1]
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t-1]
    observation : [n_dim_obs] array
        observation at time t.  If `observation` is a masked array and any of
        its values are masked, the observation will be ignored.

    Returns
    -------
    kalman_gain : [n_dim_state, n_dim_obs] array
        Kalman gain matrix for time t
    corrected_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t]
    corrected_state_covariance : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t]
    """
    if observation is not None:
        predicted_observation_mean = observation_matrix @ predicted_state_mean + observation_offset

        predicted_observation_covariance = (
            torch.mm(observation_matrix, torch.mm(predicted_state_covariance, observation_matrix.T))
            + observation_covariance
        )
        kalman_gain = torch.mm(
            predicted_state_covariance,
            torch.mm(
                observation_matrix.T, torch.linalg.pinv(predicted_observation_covariance)
            ),  # the inverse is taking long
        )

        corrected_state_mean = predicted_state_mean + kalman_gain @ (observation - predicted_observation_mean)
        corrected_state_covariance = predicted_state_covariance - torch.mm(
            kalman_gain, torch.mm(observation_matrix, predicted_state_covariance)
        )
    else:
        n_dim_state = predicted_state_covariance.shape[0]
        n_dim_obs = observation_matrix.shape[0]
        kalman_gain = torch.zeros((n_dim_state, n_dim_obs))

        corrected_state_mean = predicted_state_mean
        corrected_state_covariance = predicted_state_covariance

    return (kalman_gain, corrected_state_mean, corrected_state_covariance)


class KalmanFilterTorch(KalmanFilter):
    def __init__(
        self,
        transition_matrices=None,
        observation_matrices=None,
        transition_covariance=None,
        observation_covariance=None,
        transition_offsets=None,
        observation_offsets=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        random_state=None,
        em_vars=["transition_covariance", "observation_covariance", "initial_state_mean", "initial_state_covariance"],
        n_dim_state=None,
        n_dim_obs=None,
        device=None,
    ):
        """Initialize Kalman Filter"""

        # determine size of state space
        n_dim_state = _determine_dimensionality(
            [
                (transition_matrices, array2d, -2),
                (transition_offsets, array1d, -1),
                (transition_covariance, array2d, -2),
                (initial_state_mean, array1d, -1),
                (initial_state_covariance, array2d, -2),
                (observation_matrices, array2d, -1),
            ],
            n_dim_state,
        )
        n_dim_obs = _determine_dimensionality(
            [
                (observation_matrices, array2d, -2),
                (observation_offsets, array1d, -1),
                (observation_covariance, array2d, -2),
            ],
            n_dim_obs,
        )

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.random_state = random_state
        self.em_vars = em_vars
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs

        self.device = device

    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults"""
        n_dim_state, n_dim_obs = self.n_dim_state, self.n_dim_obs

        arguments = get_params(self)
        defaults = {
            "transition_matrices": torch.eye(n_dim_state).to(self.device),
            "transition_offsets": torch.zeros(n_dim_state).to(self.device),
            "transition_covariance": torch.eye(n_dim_state).to(self.device),
            "observation_matrices": torch.eye(n_dim_obs, n_dim_state).to(self.device),
            "observation_offsets": torch.zeros(n_dim_obs).to(self.device),
            "observation_covariance": torch.eye(n_dim_obs).to(self.device),
            "initial_state_mean": torch.zeros(n_dim_state).to(self.device),
            "initial_state_covariance": torch.eye(n_dim_state).to(self.device),
            "random_state": 0,
            "em_vars": [
                "transition_covariance",
                "observation_covariance",
                "initial_state_mean",
                "initial_state_covariance",
            ],
        }
        converters = {
            "transition_matrices": array2d,
            "transition_offsets": array1d,
            "transition_covariance": array2d,
            "observation_matrices": array2d,
            "observation_offsets": array1d,
            "observation_covariance": array2d,
            "initial_state_mean": array1d,
            "initial_state_covariance": array2d,
            "random_state": check_random_state,
            "n_dim_state": int,
            "n_dim_obs": int,
            "em_vars": lambda x: x,
        }

        parameters = preprocess_arguments([arguments, defaults], converters)

        return (
            parameters["transition_matrices"],
            parameters["transition_offsets"],
            parameters["transition_covariance"],
            parameters["observation_matrices"],
            parameters["observation_offsets"],
            parameters["observation_covariance"],
            parameters["initial_state_mean"],
            parameters["initial_state_covariance"],
        )

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_matrix=None,
        transition_offset=None,
        transition_covariance=None,
        observation_matrix=None,
        observation_offset=None,
        observation_covariance=None,
    ):
        r"""Update a Kalman Filter state estimate

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Parameters
        ----------
        filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t given observations from times
            [1...t]
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t given observations from
            times [1...t]
        observation : [n_dim_obs] array or None
            observation from time t+1.  If `observation` is a masked array and
            any of `observation`'s components are masked or if `observation` is
            None, then `observation` will be treated as a missing observation.
        transition_matrix : optional, [n_dim_state, n_dim_state] array
            state transition matrix from time t to t+1.  If unspecified,
            `self.transition_matrices` will be used.
        transition_offset : optional, [n_dim_state] array
            state offset for transition from time t to t+1.  If unspecified,
            `self.transition_offset` will be used.
        transition_covariance : optional, [n_dim_state, n_dim_state] array
            state transition covariance from time t to t+1.  If unspecified,
            `self.transition_covariance` will be used.
        observation_matrix : optional, [n_dim_obs, n_dim_state] array
            observation matrix at time t+1.  If unspecified,
            `self.observation_matrices` will be used.
        observation_offset : optional, [n_dim_obs] array
            observation offset at time t+1.  If unspecified,
            `self.observation_offset` will be used.
        observation_covariance : optional, [n_dim_obs, n_dim_obs] array
            observation covariance at time t+1.  If unspecified,
            `self.observation_covariance` will be used.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t+1 given observations from times
            [1...t+1]
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t+1 given observations
            from times [1...t+1]
        """
        # initialize matrices
        (
            transition_matrices,
            transition_offsets,
            transition_cov,
            observation_matrices,
            observation_offsets,
            observation_cov,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        transition_offset = _arg_or_default(transition_offset, transition_offsets, 1, "transition_offset")
        observation_offset = _arg_or_default(observation_offset, observation_offsets, 1, "observation_offset")
        transition_matrix = _arg_or_default(transition_matrix, transition_matrices, 2, "transition_matrix")
        observation_matrix = _arg_or_default(observation_matrix, observation_matrices, 2, "observation_matrix")
        transition_covariance = _arg_or_default(transition_covariance, transition_cov, 2, "transition_covariance")
        observation_covariance = _arg_or_default(observation_covariance, observation_cov, 2, "observation_covariance")

        # Make a masked observation if necessary
        # if observation is None:
        #     n_dim_obs = observation_covariance.shape[0]
        #     observation = np.ma.array(np.zeros(n_dim_obs))
        #     observation.mask = True
        # else:
        #     observation = np.ma.asarray(observation)

        predicted_state_mean, predicted_state_covariance = _filter_predict(
            transition_matrix, transition_covariance, transition_offset, filtered_state_mean, filtered_state_covariance
        )
        (_, next_filtered_state_mean, next_filtered_state_covariance) = _filter_correct(
            observation_matrix,
            observation_covariance,
            observation_offset,
            predicted_state_mean,
            predicted_state_covariance,
            observation,
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)
