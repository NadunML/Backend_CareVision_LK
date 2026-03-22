import urllib.request

print("Downloading AI Mask Model (Please wait...)")
url = "https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model"
# මෙතන නම .h5 විදිහට මාරු කළා!
urllib.request.urlretrieve(url, "mask_detector.h5") 
print("✅ Download Complete! You can see 'mask_detector.h5' in your folder.")