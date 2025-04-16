from datasets import load_dataset

# Enter the path to the dataset you want to load
main_data = load_dataset("whopriyam2/SUWMIT-dataset", data_files="length_20/train_elife_BioBERT.csv")

print (main_data["train"][0])
