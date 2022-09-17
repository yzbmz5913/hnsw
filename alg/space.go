package hnsw

import (
	"fmt"
	"math"
)

type Point struct {
	data []float64
	conn map[int][]*Point // connections at each level
}

func (p *Point) String() string {
	return fmt.Sprintf("%v", p.data)
}

func NewPoint(cord []float64) *Point {
	return &Point{
		data: cord,
		conn: map[int][]*Point{},
	}
}

type Space interface {
	Dim() int
	GetDist(p1, p2 *Point) float64
}

// l2Space use euclidean distance
type l2Space struct {
	dim int
}

func (l *l2Space) Dim() int {
	return l.dim
}

func (l *l2Space) GetDist(p1, p2 *Point) float64 {
	var sum float64
	for i := 0; i < l.Dim(); i++ {
		sum += (p1.data[i] - p2.data[i]) * (p1.data[i] - p2.data[i])
	}
	return math.Sqrt(sum)
}

func NewL2Space(dim int) Space {
	return &l2Space{dim: dim}
}

// csSpace use cosine similarity distance
type csSpace struct {
	dim int
}

func (c *csSpace) Dim() int {
	return c.dim
}

func (c *csSpace) GetDist(p1, p2 *Point) float64 {
	var ip, sqr1, sqr2 float64
	for i := 0; i < c.Dim(); i++ {
		ip += p1.data[i] * p2.data[i]
		sqr1 += p1.data[i] * p1.data[i]
		sqr2 += p2.data[i] * p2.data[i]
	}
	return 1 - ip/math.Sqrt(sqr1*sqr2)
}
