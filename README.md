# MIAI - AI Medical Imaging Analysis System

This project aims to build an AI-driven medical imaging analysis system based on deep learning, designed to assist doctors in diagnosing diseases through automated image detection. We have poured our passion and dedication into creating an efficient, precise, and user-friendly system that brings positive transformation to the healthcare field. The system mainly focuses on two application scenarios:  
- **Breast Cancer Detection**: Uses Convolutional Neural Networks (CNNs) to automatically detect breast cancer in ultrasound images, accurately highlighting suspicious regions and improving diagnostic accuracy and efficiency.  
- **COVID-19 Detection**: Utilizes a U-Net model to segment chest X-ray images, automatically marking infected areas to assist doctors in rapidly assessing the severity of the disease.

The system adopts a modular design with clear responsibilities for each module. All configurations are centrally managed in the global configuration file `config.py` located at the project root, ensuring unified system parameter management and ease of maintenance.

---

## Directory Structure

```
MIAI/
├── config.py                # Global configuration file (DB connections, data source URLs, scheduling strategies, log levels, etc.)
├── data/                    
│   ├── ingestion/           
│   │   ├── url_data_fetcher.py    # Fetches medical imaging data from a specified URL
│   │   └── image_preprocessor.py  # Image preprocessing: format conversion, denoising, normalization
│   └── annotation/                
│       ├── labeling_tool.py       # Data annotation tool (interactive or semi-automatic)
│       ├── annotation_manager.py  # Manages annotated data & version control
│       └── export_format.py       # Converts annotated data to standard formats (e.g., COCO, Pascal VOC)
├── model_training/
│   ├── breast_cancer/             
│   │   ├── cnn_model.py           # CNN model definition for breast cancer detection
│   │   ├── train.py               # Model training script
│   │   ├── data_augmentation.py   # Data augmentation & preprocessing
│   │   └── evaluation.py          # Model evaluation (accuracy, recall, etc.)
│   └── covid19_segmentation/      
│       ├── unet_model.py          # U-Net model definition for COVID-19 segmentation
│       ├── train.py               # Segmentation model training script
│       ├── segmentation_postprocess.py  # Post-processing of segmentation results
│       └── metrics.py             # Segmentation evaluation metrics (Dice, IoU, etc.)
├── model_management/
│   ├── version_control.py         # Model version management
│   ├── performance_monitor.py     # Model performance monitoring
│   └── hyperparameter_tuning.py   # Automated hyperparameter tuning
├── inference/
│   ├── breast_cancer_inference.py  # Inference logic for breast cancer detection
│   ├── covid19_inference.py        # Inference logic for COVID-19 segmentation
│   └── inference_manager.py        # Inference task scheduling & resource allocation
├── diagnosis/
│   ├── result_visualizer.py       # Converts model outputs into intuitive graphics/heatmaps
│   ├── report_generator.py        # Automatically generates diagnostic reports (results, confidence, etc.)
│   └── export_tools.py            # Exports reports in PDF/image formats
├── backend/
│   ├── views.py                   # REST API endpoints (image upload, query, diagnostic results, etc.)
│   ├── urls.py                    # API routing configuration
│   ├── tasks.py                   # Celery async tasks (data ingestion, model training, inference, etc.)
│   └── admin.py                   # Backend admin interface (user, permission, and task management)
├── frontend/
│   ├── app.py                     # Frontend application entry point
│   ├── layout.py                  # Page layout definition (image display, results display, report listing, etc.)
│   ├── callbacks.py               # Frontend interaction logic (upload, real-time refresh, feedback, etc.)
│   └── components/                # Custom components (image viewers, charts, report readers)
├── docker_composed/
│   ├── docker-compose.yml         # Docker Compose configuration for containerized deployment of all services
│   └── README_deploy.md           # Deployment documentation
├── requirements/
│   ├── requirements.txt           # Python dependency list
│   └── additional_requirements.md # Other system dependency documentation
└── docs/
    └── README.md                  # Project documentation (this file)
```

---

## System Architecture and Data Flow

1. **Data Ingestion and Annotation**  
   - **Data Ingestion**:  
     - The `data/ingestion/url_data_fetcher.py` module fetches medical imaging data from a specified URL, eliminating the need to connect directly to medical devices.  
     - The `data/ingestion/image_preprocessor.py` module handles image preprocessing (format conversion, denoising, normalization) to ensure the data is suitable for model training.  
   - **Data Annotation**:  
     - Data annotators use `data/annotation/labeling_tool.py` to interactively or semi-automatically annotate images.  
     - `data/annotation/annotation_manager.py` manages the annotated data and versions, while `export_format.py` converts them to standard formats (COCO, Pascal VOC, etc.) for model training.

2. **Model Training and Management**  
   - The directories `model_training/breast_cancer` and `model_training/covid19_segmentation` contain training pipelines for breast cancer detection and COVID-19 segmentation, respectively, using preprocessed and annotated data.  
   - The `model_management` module monitors model versions, performance, and hyperparameters, ensuring continuous optimization and reliable operation.

3. **Real-time Inference and Diagnosis**  
   - Doctors upload images via the frontend, and the backend invokes the corresponding model (breast cancer detection or COVID-19 segmentation) in the `inference` layer for real-time or batch inference.  
   - The `diagnosis` layer processes the model outputs to generate intuitive graphical representations and comprehensive diagnostic reports, which can be easily reviewed and archived.

4. **Backend and Frontend Services**  
   - The backend (Django or FastAPI with Celery) provides REST APIs, task scheduling, and system administration, supporting asynchronous execution of large-scale tasks (e.g., data ingestion, model training, batch inference).  
   - The frontend offers a user-friendly GUI for uploading images, viewing real-time diagnostic results, and accessing interactive reports, greatly enhancing the doctor’s workflow.

5. **Global Configuration Management**  
   - All modules in the system read shared configuration parameters from the `config.py` file at the project root, including database connections, data source URLs, task scheduling, log levels, etc. This ensures centralized management and simplifies maintenance.

6. **Deployment and Integration**  
   - The system is containerized using Docker and Docker Compose, enabling one-click startup and simplified environment setup.  
   - Integration with hospital PACS/EHR systems is supported, providing automatic data transfer and archiving while complying with HIPAA/GDPR standards for data security and privacy.

---

## Usage Instructions

1. **Configuration Management**  
   - Modify the `config.py` file in the project root to update global parameters such as data source URLs, database connections, log levels, and task scheduling.

2. **Data Ingestion and Annotation**  
   - Run `data/ingestion/url_data_fetcher.py` to fetch medical imaging data from the specified URL and use `data/ingestion/image_preprocessor.py` for preprocessing.  
   - Use `data/annotation/labeling_tool.py` to annotate images, creating a standardized dataset for model training.

3. **Model Training**  
   - Navigate to `model_training/breast_cancer` or `model_training/covid19_segmentation` and run `train.py` to train the respective models.  
   - The `model_management` module can be used to monitor model versions, performance metrics, and to perform hyperparameter tuning.

4. **Real-time Inference and Diagnosis**  
   - The backend receives uploaded images through REST APIs and automatically calls the relevant inference module in `inference` for real-time or batch processing.  
   - Diagnostic reports are generated by `diagnosis/report_generator.py` and can be downloaded or printed through the frontend interface.

5. **Deployment**  
   - Deploy the system using the `docker_composed/docker-compose.yml` file. For detailed instructions, refer to `docker_composed/README_deploy.md`.

---

## Summary

The MIAI system is designed with a modular architecture, covering data ingestion, model training, real-time inference, backend/frontend services, and deployment. All configurations are in `config.py`, ensuring centralized management. 
