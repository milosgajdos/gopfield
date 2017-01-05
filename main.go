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
	// path to data directory
	datadir string
	// path to input data
	input string
	// path to retored data
	output string
	// max number of iterations
	maxiters int
	// equilibrium iterations
	eqiters int
)

func init() {
	flag.StringVar(&datadir, "datadir", "", "Path to data directory")
	flag.StringVar(&input, "input", "", "Path to input data")
	flag.StringVar(&output, "output", "", "Path to output data")
	flag.IntVar(&maxiters, "maxiters", 0, "Max number of Hopfield net iterations")
	flag.IntVar(&eqiters, "eqiters", 0, "Number of Hopfield net equilibrium iterations")
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

func SaveImage(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	switch filepath.Ext(path) {
	case "jpeg":
		return jpeg.Encode(f, img, &jpeg.Options{100})
	case "png":
		return png.Encode(f, img)
	default:
		return png.Encode(f, img)
	}

	return nil
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
		pattern, err := hopfield.Image2Pattern(img)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		patterns[i] = pattern
	}

	// set size to the length of the read pattern
	n, err := hopfield.NewNet(len(patterns[0]))
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// store patterns in Hopfield network
	for i := range patterns {
		if err := n.Store(patterns[i]); err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
	}

	var resPattern hopfield.Pattern
	// if no input is passed it we will generate our own noisy data
	if input == "" {
		// add some noise into one of the patterns
		noisyPattern := hopfield.AddNoise(patterns[0], 50)
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
		pattern, err := hopfield.Image2Pattern(img)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
			os.Exit(1)
		}
		resPattern = pattern
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
