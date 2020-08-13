## PROJECT: TECHX_2020_OURVISON

### Description:
This is the repo for the hackathon project OUR_VISION created in Aug 2020. 
The project aims for build simulated experiences through AR/VR for users to reconstruct visions from some special groups, like colorblinds.

### Repo Structure
```bash
.
├── .idea: auto generated folder by pycharm
├── src: contains project source code
│   ├── static: dynamic files like css, js for flask
│   ├── templates: html files for flask
│   ├── Server.py: flask server
│   ├── colorblindProcessor.py: converts normal image to color blind images 
│   ├── cv_camera: read from wifi webcam and feed to processors
│   └── imageReader: not implemented yet. plan to act as reader after image finishes processing.
├── venv: virtual environemnt folder for the project

```

### How To Run The Repo
- AR Track:
    - Prerequisites: Used EpocCam to connect ios device as a webcam [at v1 not trying to implement this streaming part].
    - After got stream input from webcam, then we can run cv_camera.py to process images

### Virtual Environments / Requirements