import prediction

ps = prediction.PredictiveSearch("./models/assign3.model.hdf5", "./data/tokenizer10000.pickle", "./data/enc_images.csv.tar.gz")
ps.predictive_search("A zebra")
