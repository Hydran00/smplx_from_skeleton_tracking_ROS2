import json
import mmap
import os

max_size = 1024 * 1024 * 10
file_path = 'test_mmap'

def read_data():
    while True:
        size = min(max_size, os.path.getsize(file_path))
        with open(file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), size)
            mm.seek(0)
            json_length_bytes = mm.read(4)
            if not json_length_bytes:
                mm.close()
                continue
            
            json_length = int.from_bytes(json_length_bytes, byteorder='little')
            json_bytes = mm.read(json_length)
            mm.close()
        
        json_bytes = json_bytes.decode("utf-8").rstrip('\0')
        
        try:
            data = json.loads(json_bytes)
            print(data)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

if __name__ == '__main__':
    read_data()
