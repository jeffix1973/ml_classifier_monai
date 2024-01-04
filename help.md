# Classification model REST API

This program runs a Flask / Gunicorn server. By sending a POST request, it triggers a PyTorch MONAI pre-trained model to predcit the image located at file_path on the same server.


## Request structure

### Call urls depends on image orientation (orientation must be known before infering the model)
> http://localhost:5000/ax
> http://localhost:5000/sag
> http://localhost:5000/cor

### header:
> Content-Type : "application/json"

### body:

> {"filepath" : "local_path_to_dicom_file"}


## Typical response
```
{​​​​​​​​​​​​​​
    "ModelName":"<model_name>"
    "Predictions": {​​​​​​​​​​​​​​
        "PredictedClass": "breast",
        "ModelOutputClasses": [
            "BRAIN",
            "BREAST",
            "LIVER",
            "PROSTATE",
            "KNEE",
            "SHOULDER",
            "FOOT",
            "CERVICAL",
            "LUMBAR"  
        ]
    }​​​​​​​​​​​​​​​​​​​​​,
    "Scores": [
        8.252662e-06,
        0.999966,
        2.4763687e-05,
        9.421578e-07
    ],
    "SeriesInstanceUID": "1.3.12.2.1107.5.2.33.37013.2010072011425882374834465.0.0.15",
    "SopInstanceUID": "1.3.12.2.1107.5.2.33.37013.2010072011503343698444262",
    "StudyInstanceUID": "1.3.46.670589.16.2.2.164.1.77.227.20100603.120302.2491590"
}

```
