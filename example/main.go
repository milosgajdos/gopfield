package main

import (
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"

	"github.com/milosgajdos83/gopfield/hopfield"
)

const (
	cliname = "gopfield"
	width   = 28
	height  = 28
)

var (
	// path to data pattern directory
	datadir string
	// path to input data
	input string
	// path to retored data
	output string
	// max number of iterations
	maxiters int
	// equilibrium iterations
	eqiters int
	// learning defines type of learning
	learning string
)

func init() {
	flag.StringVar(&datadir, "datadir", "", "Path to data pattern directory")
	flag.StringVar(&input, "input", "", "Path to input data pattern")
	flag.StringVar(&output, "output", "", "Path to output data pattern")
	flag.IntVar(&maxiters, "maxiters", 0, "Max number of Hopfield net run iterations")
	flag.IntVar(&eqiters, "eqiters", 0, "Number of Hopfield net equilibrium iterations")
	flag.StringVar(&learning, "learning", "hebbian", "Type of Hopfield Network learning: hebbian or storkey")
}

// parseCliFlags parses command line args
func parseCliFlags() error {
	flag.Parse()

	if datadir == "" {
		return fmt.Errorf("Invalid path to data directory supplied: %s\n", datadir)
	}

	if output == "" {
		return fmt.Errorf("Invalid output path supplied: %s\n", output)
	}

	if maxiters <= 0 {
		return fmt.Errorf("Invalid max number of iterations: %d\n", maxiters)
	}

	if eqiters <= 0 {
		return fmt.Errorf("Invalid number of equilibrium iterations: %d\n", eqiters)
	}

	return nil
}

// ReadImage reads an image file in path and returns it as image.Image or fails with error
func ReadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	return img, nil
}

// SaveImage saves img image in path or fails with error
func SaveImage(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	switch filepath.Ext(path) {
	case "jpeg":
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	case "png":
		return png.Encode(f, img)
	}

	return fmt.Errorf("Unsupported image format: %s\n", filepath.Ext(path))
}

func main() {
	// exit if incorrect cli params were passed in
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// read datadir from supplied directory
	files, err := ioutil.ReadDir(datadir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	if len(files) == 0 {
		fmt.Fprintf(os.Stderr, "\nERROR: No patterns found in %s\n", datadir)
		os.Exit(1)
	}

	// read in Hopfield network patterns from data files in datadir
	patterns := make([]hopfield.Pattern, len(files))
	for i := range files {
		img, err := ReadImage(path.Join(datadir, files[i].Name()))
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		// convert Image to Pattern
		patterns[i] = hopfield.Image2Pattern(img)
	}

	// Create new Hopfield Network and set its size to the length of the read pattern
	n, err := hopfield.NewNet(len(patterns[0]))
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// store patterns in Hopfield network
	if err := n.Store(patterns, learning); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	var resPattern hopfield.Pattern
	// if no input is passed it we will generate our own noisy data
	if input == "" {
		// add some noise into one of the patterns
		noisyPattern := hopfield.AddNoise(patterns[1], 20)
		//encode pattern into Gray Image
		img := hopfield.Pattern2Image(noisyPattern, image.Rect(0, 0, width, height))
		// save the noisy image for reference
		if err := SaveImage("noisy.png", img); err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		resPattern = noisyPattern
	} else {
		img, err := ReadImage(input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		// convert Image to Pattern
		resPattern = hopfield.Image2Pattern(img)
	}

	// restore image from Hopfield network
	res, err := n.Restore(resPattern, maxiters, eqiters)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// render the restored image
	img := hopfield.Pattern2Image(res, image.Rect(0, 0, width, height))
	if err := SaveImage(output, img); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
}
