# VVSG Profile Clustering

## Overview

This project is developed for VVSG (Association of Flemish Municipalities) to generate standard synthetic load profiles for public buildings in Flanders. The load profiles are developed using smart meter data. The project also includes a clustering algorithm that picks the most fitting cluster based on a set of user inputs.

## Features

- Load a pre-trained model and a scaler from disk.
- User inputs for building type and yearly electricity consumption.
- Advanced options for specifying the fraction of evening/morning and weekend/weekday load.
- Based on the user inputs, the algorithm picks the most fitting cluster and generates a standard synthetic load profile.

## Usage

The application is built with Streamlit and can be run in a web browser. Here's a brief overview of the user interface:

1. The user is asked to input their yearly consumption in kWh, which is then scaled using the loaded scaler.
2. The user is asked to select the type of building from a dropdown list.
3. The user can specify more details about the building usage in the "Advanced options". Inside the "Advanced options", the user can specify the usage of the building in the evening and on weekends using sliders. Depending on the slider value, a message about the building usage is displayed.

## Installation

To run the application, you need to have Python installed on your machine. You also need to install the required Python packages. You can install the packages using the following command:

```bash
pip install -r requirements.txt
```

```bash
streamlit run social_clusters.py
```

After installing the packages, you can run the application using the following command:

## Contributing

Contributions are welcome. Please make sure to follow the project's contribution guidelines.

## License

This project is licensed under the terms of the MIT license.