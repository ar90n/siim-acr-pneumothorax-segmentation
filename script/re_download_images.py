"""Script to download all instances in a DICOM Store."""
import os
import posixpath
from pathlib import Path
from concurrent import futures
from retrying import retry
import google.auth
from google.auth.transport.requests import AuthorizedSession

# URL of CHC API
CHC_API_URL = 'https://healthcare.googleapis.com/v1beta1'
PROJECT_ID = 'kaggle-siim-healthcare'
REGION = 'us-central1'
DATASET_ID = 'siim-pneumothorax'
TRAIN_DICOM_STORE_ID = 'dicom-images-train'
TEST_DICOM_STORE_ID = 'dicom-images-test'


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def download_instance(dicom_web_url, dicom_store_id, study_uid, series_uid,
                      instance_uid, credentials):
    """Downloads a DICOM instance and saves it under the current folder."""
    instance_url = posixpath.join(dicom_web_url, 'studies', study_uid, 'series',
                                  series_uid, 'instances', instance_uid)
    authed_session = AuthorizedSession(credentials)
    response = authed_session.get(
        instance_url, headers={'Accept': 'application/dicom; transfer-syntax=*'})
    file_path = posixpath.join(dicom_store_id, study_uid, series_uid,
                               instance_uid)
    filename = '%s.dcm' % file_path
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        f.write(response.content)


def download_all_failed_instances(dicom_store_id, credentials):
    """Downloads all DICOM instances in the specified DICOM store."""
    # Get a list of all failed instances.
    content = []
    max_failed_data_size = 1024
    for study in Path(dicom_store_id).glob('*'):
        study_uid = study.name
        for series in study.glob('*'):
            series_uid = series.name
            for instance in series.glob('*'):
                if max_failed_data_size < instance.stat().st_size:
                    continue

                instance_uid = instance.stem
                content.append((study_uid, series_uid, instance_uid))

    dicom_web_url = posixpath.join(CHC_API_URL, 'projects', PROJECT_ID,
                                   'locations', REGION, 'datasets', DATASET_ID,
                                   'dicomStores', dicom_store_id, 'dicomWeb')
    with futures.ThreadPoolExecutor() as executor:
        future_to_study_uid = {}
        for study_uid, series_uid, instance_uid in content:
            future = executor.submit(download_instance, dicom_web_url, dicom_store_id,
                                     study_uid, series_uid, instance_uid, credentials)
            future_to_study_uid[future] = study_uid
        processed_count = 0
        for future in futures.as_completed(future_to_study_uid):
            try:
                future.result()
                processed_count += 1
                if not processed_count % 100 or processed_count == len(content):
                    print('Processed instance %d out of %d' %
                          (processed_count, len(content)))
            except Exception as e:
                print('Failed to download a study. UID: %s \n exception: %s' %
                      (future_to_study_uid[future], e))


def main(argv=None):
    credentials, _ = google.auth.default()
    print('Downloading all failed instances in %s DICOM store' % TRAIN_DICOM_STORE_ID)
    download_all_failed_instances(TRAIN_DICOM_STORE_ID, credentials)
    print('Downloading all failed_instances in %s DICOM store' % TEST_DICOM_STORE_ID)
    download_all_failed_instances(TEST_DICOM_STORE_ID, credentials)


main()
