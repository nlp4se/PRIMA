import logging

logger = logging.getLogger(__name__)


def get_app_id_from_filename(filename: str) -> str:
    app_id = filename.replace('.json', '')
    app_id = app_id.replace('/', '_').replace('\\', '_')

    return app_id


