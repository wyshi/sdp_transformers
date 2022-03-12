from allennlp.predictors.predictor import Predictor
from spacy.language import Language
from spacy.tokens import Doc


@Language.factory(
    "srl",
    default_config={
        "model_path": "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    },
)
def create_srl_component(nlp: Language, name: str, model_path: str):
    return SRLComponent(nlp, model_path)


class SRLComponent:
    def __init__(self, nlp: Language, model_path: str):
        if not Doc.has_extension("srl"):
            Doc.set_extension("srl", default=None)
        self.predictor = Predictor.from_path(model_path)

    def __call__(self, doc: Doc):
        predictions = self.predictor.predict(sentence=doc.text)
        doc._.srl = predictions
        return doc


if __name__ == "__main__":
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("srl")
    doc = nlp(
        "Did I already tell you I'm getting a divorce?"
        # ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
    )
    import pdb

    pdb.set_trace()
    print(doc._.srl)
