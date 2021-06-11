from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1CLdiHuNq_QNOcv6fTs8HlOnlScE9RNCO',
                                    dest_path='./sql/data.db', overwrite=True, showsize=True)
