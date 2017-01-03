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
	// max number of iterations
	maxiters int
	// equilibrium iterations
	eqiters int
)

func init() {
	flag.IntVar(&size, "size", 0, "Size of Hopfield network")
	flag.IntVar(&maxiters, "maxiters", 0, "Max number of Hopfield net iterations")
	flag.IntVar(&eqiters, "eqiters", 0, "Number of Hopfield net equilibrium iterations")
}

// parseCliFlags parses command line args
func parseCliFlags() error {
	flag.Parse()

	if size <= 0 {
		return fmt.Errorf("Invalid size supplied: %d", size)
	}

	if maxiters <= 0 {
		return fmt.Errorf("Invalid max number of iterations: %d\n", maxiters)
	}

	if eqiters <= 0 {
		return fmt.Errorf("Invalid number of equilibrium iterations: %d\n", eqiters)
	}

	return nil
}

func main() {
	// exit if incorrect cli params were passed in
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	n, err := hopfield.NewNet(size)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	pattern := hopfield.Encode([]float64{0.0, 1.0, 1.0, 0.0, 1.0})
	fmt.Println("storing", pattern)
	if err := n.Store(pattern); err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	pattern = hopfield.Encode([]float64{1.0, 0.0, 1.0, 0.0, 1.0})
	fmt.Println("storing", pattern)
	if err := n.Store(pattern); err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	w := n.Weights()
	fw := mat64.Formatted(w, mat64.Prefix(" "))
	fmt.Printf("Hopfield weights:\n %v\n\n", fw)

	pattern = hopfield.Encode([]float64{1.0, 0.0, 1.0, 1.0, 1.0})
	fmt.Println("restoring", pattern)
	res, err := n.Restore(pattern, maxiters, eqiters)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}
	fmt.Println("Restored", res)
}
