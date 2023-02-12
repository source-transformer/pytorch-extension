package main

// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu -L/workspaces/df-gotch/omg-pytorch-linux-arm64/torch/lib
// #cgo LDFLAGS: -L/usr/local/cuda/lib64
// #cgo CFLAGS: -I/workspaces/df-gotch/omg-pytorch-linux-arm64/torch/ -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=0
// #cgo CFLAGS: -I/usr/local/cuda/include -I/workspaces/df-gotch/omg-pytorch-linux-arm64/torch/include
// #cgo CXXFLAGS: -std=c++14 -I${SRCDIR} -g -O3
// #cgo CXXFLAGS: -I/workspaces/df-gotch/omg-pytorch-linux-arm64/torch/lib -I/workspaces/df-gotch/omg-pytorch-linux-arm64/torch/include -I/workspaces/df-gotch/omg-pytorch-linux-arm64/torch/include/torch/csrc/api/include
import "C"
import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

// pulled out third line
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -lc10_cuda -ltorch_cuda

const (
	LearningRate float64 = 0.01
	HiddenSize   int64   = 256
	SeqLen       int64   = 180
	BatchSize    int64   = 256
	Epochs       int     = 3
	SamplingLen  int64   = 1024
)

func sample(data *ts.TextData, lstm *nn.LSTM, linear *nn.Linear, device gotch.Device) string {
	labels := data.Labels()
	inState := lstm.ZeroState(1)
	lastLabel := int64(0)
	var runes []rune

	for i := 0; i < int(SamplingLen); i++ {
		input := ts.MustZeros([]int64{1, labels}, gotch.Float, device)
		// NOTE. `Narrow` creates tensor that shares same storage
		inputView := input.MustNarrow(1, lastLabel, 1, false)
		inputView.MustFill_(ts.FloatScalar(1.0))

		state := lstm.Step(input, inState)

		// 1. Delete inState tensors (from C land memory)
		inState.(*nn.LSTMState).Tensor1.MustDrop()
		inState.(*nn.LSTMState).Tensor2.MustDrop()
		// 2. Then update with current state
		inState = state
		// 3. Delete intermediate tensors
		input.MustDrop()
		inputView.MustDrop()

		forwardTs := linear.Forward(state.(*nn.LSTMState).H()).MustSqueezeDim(0, true).MustSoftmax(-1, gotch.Float, true)
		sampledY := forwardTs.MustMultinomial(1, false, true)
		lastLabel = sampledY.Int64Values()[0]
		sampledY.MustDrop()
		char := data.LabelForChar(lastLabel)

		runes = append(runes, char)
	}

	// Delete the last state
	inState.(*nn.LSTMState).Tensor1.MustDrop()
	inState.(*nn.LSTMState).Tensor2.MustDrop()

	return string(runes)
}

func main() {
	device := gotch.CudaIfAvailable()

	vs := nn.NewVarStore(device)
	//data, err := ts.NewTextData("../../data/char-rnn/input.txt")
	data, err := ts.NewTextData("../../data/char-rnn/input-500.txt")
	if err != nil {
		panic(err)
	}

	labels := data.Labels()
	fmt.Printf("Dataset loaded, %v labels\n", labels)

	lstm := nn.NewLSTM(vs.Root(), labels, HiddenSize, nn.DefaultRNNConfig())
	linear := nn.NewLinear(vs.Root(), HiddenSize, labels, nn.DefaultLinearConfig())

	optConfig := nn.DefaultAdamConfig()
	opt, err := optConfig.Build(vs, LearningRate)
	if err != nil {
		log.Fatal(err)
	}

	for epoch := 1; epoch <= Epochs; epoch++ {
		sumLoss := 0.0
		cntLoss := 0.0

		dataIter := data.IterShuffle(SeqLen+1, BatchSize)

		batchCount := 0
		for {
			batchTs, ok := dataIter.Next()
			if !ok {
				break
			}

			batchNarrow := batchTs.MustNarrow(1, 0, SeqLen, false)
			xsOnehot := batchNarrow.Onehot(labels).MustTo(device, true) // [256, 180, 65]
			batchNarrow.MustDrop()

			ys := batchTs.MustNarrow(1, 1, SeqLen, true).MustTotype(gotch.Int64, true).MustTo(device, true).MustView([]int64{BatchSize * SeqLen}, true)

			lstmOut, outState := lstm.Seq(xsOnehot)
			// NOTE. Although outState will not be used. There a hidden memory usage
			// on C land memory that is needed to free up. Don't use `_`
			outState.(*nn.LSTMState).Tensor1.MustDrop()
			outState.(*nn.LSTMState).Tensor2.MustDrop()
			xsOnehot.MustDrop()

			logits := linear.Forward(lstmOut)
			lstmOut.MustDrop()
			lossView := logits.MustView([]int64{BatchSize * SeqLen, labels}, true)

			loss := lossView.CrossEntropyForLogits(ys)
			ys.MustDrop()
			lossView.MustDrop()

			opt.BackwardStepClip(loss, 0.5)
			sumLoss += loss.Float64Values()[0]
			cntLoss += 1.0
			loss.MustDrop()

			batchCount++
			if batchCount%500 == 0 {
				fmt.Printf("Epoch %v - Batch %v \n", epoch, batchCount)
			}

			fmt.Printf("dataIter: progress: %v\n", dataIter.Progress())
		} // infinite for-loop

		sampleStr := sample(data, lstm, linear, device)
		fmt.Printf("Epoch %v - Loss: %v \n", epoch, sumLoss/cntLoss)
		fmt.Println(sampleStr)

		dataIter.Data.MustDrop()
		dataIter.Indexes.MustDrop()
	}
}
