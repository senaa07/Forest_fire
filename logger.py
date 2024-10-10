import datetime


class logger:
    def __init__(self):
        self.log_entries = []

    def log_event(self, frame_number, coordinates):
        """Logs an event with the given type and data."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "coordinates": f"({coordinates[0]}, {coordinates[1]}, {coordinates[2]}, {coordinates[3]})",
        }
        self.log_entries.append(log_entry)

    def _write_logs_to_file(self):
        with open("fire_detection_log.txt", "w") as file:
            for entry in self.log_entries:
                file.write(
                    f"{entry['timestamp']} - Frame {entry['frame_number']}: Fire detected at {entry['coordinates']}\n"
                )
        print("Logs have been written to fire_detection_log.txt.")
