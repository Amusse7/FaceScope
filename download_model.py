import urllib.request
import os
import ssl
import certifi

def download_model():
    # Model URL - official OpenCV model
    url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    # Local path
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    print("[INFO] Downloading face detection model...")
    try:
        # Create SSL context using certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Remove existing corrupted file
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Download new model with SSL context
        urllib.request.urlretrieve(
            url, 
            model_path,
            context=ssl_context
        )
        
        # Verify file size
        size = os.path.getsize(model_path)
        print(f"[INFO] Downloaded model size: {size:,} bytes")
        
        if size < 10000000:  # Model should be ~10MB
            raise Exception("Downloaded model file is too small")
            
        print("[INFO] Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Download failed: {str(e)}")
        # Alternative download method if the first fails
        try:
            print("[INFO] Trying alternative download method...")
            cmd = f"curl -L 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel' -o {model_path}"
            os.system(cmd)
            
            size = os.path.getsize(model_path)
            print(f"[INFO] Downloaded model size: {size:,} bytes")
            
            if size < 10000000:
                raise Exception("Downloaded model file is too small")
                
            print("[INFO] Model downloaded successfully using curl")
            return True
        except Exception as e2:
            print(f"[ERROR] Alternative download also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    # First, ensure certifi is installed
    try:
        import certifi
        
    except ImportError:
        print("[INFO] Installing required package: certifi")
        os.system("pip3 install certifi")
        import certifi
    
    download_model()