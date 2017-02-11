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
	// training defines type of learning
	training string
)

func init() {
	flag.StringVar(&datadir, "datadir", "", "Path to data pattern directory")
	flag.StringVar(&input, "input", "", "Path to input data pattern")
	flag.StringVar(&output, "output", "", "Path to output data pattern")
	flag.IntVar(&maxiters, "maxiters", 0, "Max number of Hopfield net run iterations")
	flag.IntVar(&eqiters, "eqiters", 0, "Number of Hopfield net equilibrium iterations")
	flag.StringVar(&training, "training", "hebbian", "Type of Hopfield Network training: hebbian or storkey")
}

// parseCliFlags parses command line args
func parseCliFlags() error {
	flag.Parse()

	if datadir == "" {
		return fmt.Errorf("invalid path to data directory supplied: %s", datadir)
	}

	if output == "" {
		return fmt.Errorf("invalid output path supplied: %s", output)
	}

	if maxiters <= 0 {
		return fmt.Errorf("invalid max number of iterations: %d", maxiters)
	}

	if eqiters <= 0 {
		return fmt.Errorf("invalid number of equilibrium iterations: %d", eqiters)
	}

	return nil
}

// readImage reads an image file in path and returns it as image.Image or fails with error
func readImage(path string) (image.Image, error) {
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

// saveImage saves img image in path or fails with error
func saveImage(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	switch filepath.Ext(path) {
	case ".jpeg":
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	case ".png":
		return png.Encode(f, img)
	}

	return fmt.Errorf("Unsupported image format: %s", filepath.Ext(path))
}

func main() {
	// exit if incorrect cli params were passed in
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %s\n", err)
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
		img, err := readImage(path.Join(datadir, files[i].Name()))
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		// convert Image to Pattern
		patterns[i] = hopfield.Image2Pattern(img)
	}

	// Create new Hopfield Network and set its size to the length of the read pattern
	n, err := hopfield.NewNet(len(patterns[0]), training)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// store patterns in Hopfield network
	if err := n.Store(patterns); err != nil {
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
		if err := saveImage("noisy.png", img); err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		resPattern = noisyPattern
	} else {
		img, err := readImage(input)
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
	if err := saveImage(output, img); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}
}
