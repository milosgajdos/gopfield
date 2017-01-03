package hopfield

// Pattern is a data pattern
type Pattern []float64

// Encode encodes pattern to a binary pattern of values: +1/-1 where non-positive items are set to -1.
// Encode modifies the passed in pattern in place and returns it.
func Encode(p Pattern) Pattern {
	for i := 0; i < len(p); i++ {
		if p[i] <= 0.0 {
			p[i] = -1.0
		} else {
			p[i] = 1.0
		}
	}

	return p
}
