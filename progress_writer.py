from pathlib import Path

class ProgressWriter:
    def __init__(self, folder='./progress', filename='progress.csv'):
        self.folder = folder
        self.filename = filename
    
    def add_entry(self, entry):
        folder_path = Path(self.folder)
        if folder_path.is_dir() == False:
            folder_path.mkdir(parents=True, exist_ok=True)
        
        file_path = str(folder_path / self.filename)
        print(f"Write statistics to '{file_path}'\n")
        with open(file_path, 'a+') as file_handle:
            if file_handle.tell() == 0:
                # Write header
                file_handle.write(",".join(entry.keys()) + "\n")
            file_handle.write(",".join(str(x) for x in entry.values()) + "\n")


