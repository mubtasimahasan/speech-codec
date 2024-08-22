# Use the base image
FROM pytorchlightning/pytorch_lightning:latest

# Set the working directory in the container
WORKDIR /app

# Install Cython first
RUN pip install --no-cache-dir Cython==3.0.8

# Install required Python packages
RUN pip install --no-cache-dir \
    einops==0.8.0 \
    argbind==0.3.9 \
    torchaudio==2.2.1 \
    soundfile==0.12.1 \
    accelerate==0.33.0 \
    munch==4.0.0 \
    librosa==0.10.2 \
    transformers==4.43.2 \
    nemo_toolkit==1.23.0 \
    descript-audio-codec==1.0.0 \
    datasets==2.20.0 \
    phonemizer==3.3.0 \
    hydra-core==1.3.2 \
    huggingface_hub==0.23.2 \
    sentencepiece==0.2.0 \
    youtokentome==1.0.6 \
    inflect==7.3.1 \
    editdistance==0.8.1 \
    lhotse==1.26.0 \
    pyannote.core==5.0.0 \
    webdataset==0.2.96 \
    pyannote.metrics==3.2.1 \
    jiwer==2.5.2 \
    wandb==0.17.7

# Install system dependencies
RUN apt-get update && apt-get install -y \
    festival \
    espeak-ng \
    mbrola \
    && rm -rf /var/lib/apt/lists/*