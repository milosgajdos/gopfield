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
)

func init() {
	flag.IntVar(&size, "size", 0, "Size of Hopfield network")
}

func parseCliFlags() error {
	flag.Parse()

	if size <= 0 {
		return fmt.Errorf("Invalid size supplied: %d", size)
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
	data := mat64.NewDense(2, 4, []float64{
		1.0, 0.0, 0.0, 1.0,
		0.0, 1.0, 1.0, 0.0})

	if err := n.Store(data); err != nil {
		fmt.Fprintf(os.Stderr, "\n ERROR: %s\n", err)
		os.Exit(1)
	}

	w := n.Weights()
	fw := mat64.Formatted(w, mat64.Prefix(" "))
	fmt.Printf("Hopfield weights:\n %v\n\n", fw)
}
