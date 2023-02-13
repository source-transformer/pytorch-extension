#include <torch/torch.h>
#include "RnnTrainer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>


const int SEQ_LENGTH = 10;
const int INPUT_SIZE = 1;
const int HIDDEN_SIZE = 32;
const int NUM_LAYERS = 1;
const int NUM_EPOCHS = 300;
const float LEARNING_RATE = 0.001;

struct RNNImpl : torch::nn::Module {
  RNNImpl()
      : rnn(torch::nn::RNNOptions(INPUT_SIZE, HIDDEN_SIZE)
                .num_layers(NUM_LAYERS)
                .batch_first(true)) {
    register_module("rnn", rnn);
  }

  torch::Tensor forward(torch::Tensor x) {
    // Reset the hidden state.
    h0 = torch::zeros({NUM_LAYERS, x.size(0), HIDDEN_SIZE});
    // apparently my version of C++ doesn't understand the following
    //auto out, hn = rnn->forward(x, h0);
    std::tuple<torch::Tensor, torch::Tensor> out_hn = rnn->forward(x, h0);
    torch::Tensor out = std::get<0>(out_hn);
    torch::Tensor hn = std::get<1>(out_hn);
    // apparently my version of C++ doesn't understand the following
    //return out[:, -1, :];
    auto last_hidden = out.select(1, out.size(1) - 1);
    return last_hidden;
  }

  torch::nn::RNN rnn;
  torch::Tensor h0;
};
TORCH_MODULE(RNN);

std::vector<torch::Tensor> load_data(const std::string &file_path) {
  std::vector<torch::Tensor> data;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    exit(1);
  }

  std::string line;
  while (std::getline(file, line)) {
    //data.emplace_back(torch::from_blob(line.data(), {1, INPUT_SIZE}).to(torch::kFloat));
    c10::IntArrayRef c10ArrayIndices = {1, INPUT_SIZE};
    char* dataCharPtr = const_cast<char*>(line.data());
    void* dataPtr = dataCharPtr;
    auto tensor = torch::from_blob(dataPtr, c10ArrayIndices).to(torch::kFloat);
    data.emplace_back(tensor);
  }

  return data;
}

using namespace std;

RnnTrainer::RnnTrainer(){

}

RnnTrainer::~RnnTrainer(){

}

/*
string RnnTrainer::loadData(string filepath){
    std::cout << "RnnTrainer::loadData()" << std::endl;

    string fileContents;
    std::ifstream myfile (filepath);
    bool openedFile = myfile.is_open();
    if (!openedFile)
    {
        return fileContents;
    }

    // TPC: inefficient - but fine for now
    string line;
    while ( getline (myfile,line) )
    {
        //cout << line << '\n';
        fileContents.append(line);
        fileContents.append("\n");
    }
    myfile.close();
  
    return fileContents;
}

void RnnTrainer::train(){
    std::cout << "RnnTrainer::train()" << std::endl;
    torch::manual_seed(1);
    //torch::
    string fileContents = loadData("../data/input-500.txt");
    //std::cout << "fileContents: " << fileContents << std::endl;
    std::cout << "fileContents: " << fileContents.size() << std::endl;
    if (fileContents.size() == 0){
        std::cout << "fileContents is empty" << std::endl;
        return;
    }

    //torch::nn::LSTM lstm(torch::nn::LSTMOptions(10, 3).num_layers(2));
    int64_t inputSize = 10;
    int64_t hiddenSize = 3;
    //torch::nn::LSTMOptions lstmOpts = torch::nn::LSTMOptions(10, 3).num_layers(2);
    torch::nn::LSTMOptions lstmOpts1(inputSize, hiddenSize);
    torch::nn::LSTMOptions lstmOpts2 = lstmOpts1.num_layers(2);
    torch::nn::LSTM lstm(lstmOpts2);
}
*/

void RnnTrainer::train(){
  // Load the data from file.
  auto data = load_data("../data/input-500.txt");
  // the following generated the exception: std::runtime_error: shape '[-1, 10, 1]' is invalid for input of size 501
  //auto x = torch::stack(data).view({-1, SEQ_LENGTH, INPUT_SIZE});
  std::cout << "data.size(): " << data.size() << std::endl;
  int firstDimension = (data.size() / SEQ_LENGTH / INPUT_SIZE) + 1;
  // The size of a tensor is determined by its number of elements, which is the product of the sizes of all dimensions. 
  //auto x = torch::stack(data).view({firstDimension, SEQ_LENGTH, INPUT_SIZE});
  torch::Tensor dataTensor = torch::stack(data);
  //auto x = dataTensor.view({firstDimension, SEQ_LENGTH, INPUT_SIZE});
  auto x = dataTensor;
  auto y = torch::randn({x.size(0), HIDDEN_SIZE});

  // Initialize the model and optimizer.
  auto model = RNN();
  torch::optim::Adam optimizer(model->parameters(), LEARNING_RATE);

  // Train the model.
  for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
    auto y_pred = model->forward(x);
    auto loss = torch::mse_loss(y_pred, y);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    if (epoch % 100 == 0) {
      std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }
  }
}

/*
// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

// The batch size for training.
const int64_t kBatchSize = 64;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 30;

// Where to find the MNIST dataset.
const char* kDataFolder = "./data";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 200;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 10;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

using namespace torch;

struct DCGANGeneratorImpl : nn::Module {
  DCGANGeneratorImpl(int kNoiseSize)
      : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                  .bias(false)),
        batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm2(128),
        conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false))
 {
   // register_module() is needed if we want to use the parameters() method later on
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv3", conv3);
   register_module("conv4", conv4);
   register_module("batch_norm1", batch_norm1);
   register_module("batch_norm2", batch_norm2);
   register_module("batch_norm3", batch_norm3);
 }

 torch::Tensor forward(torch::Tensor x) {
   x = torch::relu(batch_norm1(conv1(x)));
   x = torch::relu(batch_norm2(conv2(x)));
   x = torch::relu(batch_norm3(conv3(x)));
   x = torch::tanh(conv4(x));
   return x;
 }

 nn::ConvTranspose2d conv1, conv2, conv3, conv4;
 nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);

int main(int argc, const char* argv[]) {
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  else{
    std::cout << "CUDA is not available! Training on CPU." << std::endl;
  }

  DCGANGenerator generator(kNoiseSize);
  generator->to(device);

  nn::Sequential discriminator(
      // Layer 1
      nn::Conv2d(
          nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 2
      nn::Conv2d(
          nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(128),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 3
      nn::Conv2d(
          nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(256),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 4
      nn::Conv2d(
          nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
      nn::Sigmoid());
  discriminator->to(device);

  // Assume the MNIST dataset is available under `kDataFolder`;
  auto dataset = torch::data::datasets::MNIST(kDataFolder)
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());
  const int64_t batches_per_epoch =
      std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));

  if (kRestoreFromCheckpoint) {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(
        discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }

  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels =
          torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real =
          torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise =
          torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake =
          torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss =
          torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      batch_index++;
      if (batch_index % kLogInterval == 0) {
        std::printf(
            "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
            epoch,
            kNumberOfEpochs,
            batch_index,
            batches_per_epoch,
            d_loss.item<float>(),
            g_loss.item<float>());
      }

      if (batch_index % kCheckpointEvery == 0) {
        // Checkpoint the model and optimizer state.
        torch::save(generator, "generator-checkpoint.pt");
        torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::save(discriminator, "discriminator-checkpoint.pt");
        torch::save(
            discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
        // Sample the generator and save the images.
        torch::Tensor samples = generator->forward(torch::randn(
            {kNumberOfSamplesPerCheckpoint, kNoiseSize, 1, 1}, device));
        torch::save(
            (samples + 1.0) / 2.0,
            torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
      }
    }
  }

  std::cout << "Training complete!" << std::endl;
}

*/

