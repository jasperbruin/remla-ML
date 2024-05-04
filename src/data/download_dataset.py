import gdown
import dvc.api

if __name__ == "__main__":
    params = dvc.api.params_show()
    gdown.download_folder(id=params["data_folder_id"], output="data/external", quiet=False)