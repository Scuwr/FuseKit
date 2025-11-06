from fusekit import Modeling, Datasets

models = [Modeling.GPT5_nano, Modeling.Claude4_Sonnet, Modeling.Gemini2p5_Flash_Lite]

for model_name in models:
    model = model_name()
    dataset = Datasets.IterableDataset()
    image_path = ["test.png"]
    question = "List the words on this image"
    label = ""
    dataset.samples = [Datasets.TextVisionSample(None, image_path, None, question, label)]

    model.evaluate(dataset)
    print(model.model_name)
    print(dataset.samples[0])