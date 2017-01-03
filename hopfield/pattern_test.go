package hopfield

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncode(t *testing.T) {
	assert := assert.New(t)

	testCases := []struct {
		p        Pattern
		expected Pattern
	}{
		{[]float64{1.0, -10.0, 0.0}, []float64{1.0, -1.0, -1.0}},
		{[]float64{1.0, 10.0}, []float64{1.0, 1.0}},
	}

	for _, tc := range testCases {
		res := Encode(tc.p)
		assert.EqualValues(tc.expected, res)
	}
}
