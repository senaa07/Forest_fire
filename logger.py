import datetime


class logger:
    def __init__(self, log_file_path):
        self.log_entries = []
        self.log_file = open(log_file_path, "w")

    def log_event(self, frame_number, coordinates):
        """Logs an event with the given type and data."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "coordinates": f"({coordinates[0]}, {coordinates[1]}, {coordinates[2]}, {coordinates[3]})",
        }
        self.log_file.write(log_entry)
        self.log_file.flush()

    def _cleanup(self):
        if self.log_file:
            self.log_file.close()
