## PROJECT: TECHX_2020_OURVISON

### Description:
This is the repo for the hackathon project OUR_VISION created in Aug 2020. 
The project aims for build simulated experiences through AR/VR for users to reconstruct visions from some special groups, like colorblinds.

### Repo Structure
``` bash
.
├── .idea: auto generated folder by pycharm
├── src: contains project source code
│   ├── static: dynamic files like css, js for flask
│   ├── templates: html files for flask
│   ├── app.py: flask server
│   ├── colorblindProcessor.py: converts normal image to color blind images (Currently not used in hackathon version)
│   ├── cv_camera: image processing module, get video stream feed from wifi device and render to html pages with flask
│   ├── cv_kernel: image processing kernels
│   ├── general_detection: image detection module
│   ├── FunctionalityTester, imageReader, histogram_trial: testing scripts 
│   ├── yolo3.cfg, yolov3.weights, coconames: yolo files for general_detection
│   ├── 123.wav, 12345.jpg, checks.png: trial objects in various trial scripts  
│   └── imageReader: not implemented yet. plan to act as reader after image finishes processing.
├── venv: virtual environemnt folder for the project

```
### How To Run The Repo
- run app.py with on extra arguments, then an html page with image updating feature and audio will be rendered.
You can access the page with localhost:5003, or check your local area network ips.
- you can switch camera index in cv_camera.py, due to time issue, we didn't make a global config file.
You need to switch values manually.

### Virtual Environments / Requirements
quite a bunch xd