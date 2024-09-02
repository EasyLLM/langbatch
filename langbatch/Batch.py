import jsonlines
import logging
import uuid

class Batch:
    def __init__(self, file):
        # OpenAI compatible file
        self._file = file
        self._id = uuid.uuid4()

    def _prepare_batch(self, requests):
        self._id = uuid.uuid4()
        file_path = f"{self._id}.jsonl"
        with jsonlines.open(file_path, mode='w') as writer:
            writer.write_all(requests)

        return file_path

    def _upload_batch_file(self):
        pass

    def start(self):
        pass

    def cancel(self):
        pass

    def status(self):
        pass

    def _get_requests(self):
        requests = []
        with jsonlines.open(self._file) as reader:
            for obj in reader:
                requests.append(obj)
        return requests

    def _validate_request(self, request):
        pass

    def _validate_requests(self):
        invalid_requests = []
        for request in self._get_requests():
           valid = self._validate_request(**request['body'])
           
           if not valid:
               invalid_requests.append(request['custom_id'])

        if len(invalid_requests) > 0:
            raise ValueError(f"Invalid requests: {invalid_requests}")
    
    # Retry on rate limit fail cases
    def _retry(self):
        pass

    def _download_results_file(self):
        pass
    
    # return results file in OpenAI compatible format
    def get_results_file(self):
        pass
    
    # return results list
    def get_results(self):
        pass
    
    