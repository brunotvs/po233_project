from google_drive_downloader import GoogleDriveDownloader as gdd
from project_utils.constants import targets_models

for target, _ in targets_models.items():
    gdd.download_file_from_google_drive(
        file_id=targets_models[target],
        dest_path=f'model/{target}.pkl',
        overwrite=True,
        showsize=True)

gdd.download_file_from_google_drive(file_id='1CLdiHuNq_QNOcv6fTs8HlOnlScE9RNCO',
                                    dest_path='sql/data.db', overwrite=True, showsize=True)
