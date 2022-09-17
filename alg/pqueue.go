package hnsw

import "fmt"

type pQueue []*Pair[float64, *Point]

func (q *pQueue) Len() int {
	return len(*q)
}

func (q *pQueue) Less(i, j int) bool {
	return (*q)[i].first < (*q)[j].first
}

func (q *pQueue) Swap(i, j int) {
	(*q)[i], (*q)[j] = (*q)[j], (*q)[i]
}

func (q *pQueue) Push(x any) {
	*q = append(*q, x.(*Pair[float64, *Point]))
}

func (q *pQueue) Pop() any {
	p := *q
	v := p[len(p)-1]
	*q = p[:len(p)-1]
	return v
}

type Pair[A, B any] struct {
	first  A
	second B
}

func NewPair[A, B any](first A, second B) *Pair[A, B] {
	return &Pair[A, B]{first, second}
}

func (p *Pair[A, B]) String() string {
	return fmt.Sprintf("(%v, %v)", p.first, p.second)
}
