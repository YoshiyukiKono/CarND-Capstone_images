# CarND-Capstone_images
TF Record generation for CarND-Capstone

### Command : json to yaml 
```bash
yq -y "." Sloth-json-format-sample.json
- annotations:
  - class: rect
    height: 60
    width: 46
    x: 346
    y: 105
  - class: rect
    height: 58
    width: 56
    x: 636
    y: 119
  class: image
  filename: image1.jpg
- class: image
  filename: image2.jpg
```