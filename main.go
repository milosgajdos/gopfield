package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/gonum/matrix/mat64"
	"github.com/milosgajdos83/gopfield/hopfield"
)

const (
	cliname = "gopfield"
)

var (
	// size of Hopfield net
	size int
	// number of iterations
	iters int
)

func init() {
	flag.IntVar(&size, "size", 0, "Size of Hopfield network")
	flag.IntVar(&iters, "iters", 0, "Number of Hopfield net iterations")
}

func parseCliFlags() error {
	flag.Parse()

	if size <= 0 {
		return fmt.Errorf("Invalid size supplied: %d", size)
	}

	if iters <= 0 {
		return fmt.Errorf("Invalid number of iterations: %d\n", iters)
	}

	return nil
}

func main() {

	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	n, err := hopfield.NewNet(size)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	// test data patterns to store in network
	//	data := mat64.NewDense(2, 4, []float64{
	//		1.0, 0.0, 0.0, 1.0,
	//		0.0, 1.0, 1.0, 0.0})
	//
	//data := []float64{1.0, 0.0, 0.0, 1.0}
	data := []float64{0.0, 1.0, 1.0, 0.0, 1.0}
	fmt.Println("storing", data)
	if err := n.Store(data); err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	data = []float64{1.0, 0.0, 1.0, 0.0, 1.0}
	fmt.Println("storing", data)
	if err := n.Store(data); err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	w := n.Weights()
	fw := mat64.Formatted(w, mat64.Prefix(" "))
	fmt.Printf("Hopfield weights:\n %v\n\n", fw)

	fmt.Println("restoring", data)
	res, err := n.Restore(data, iters)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}
	fmt.Println("Restored", res)
}
