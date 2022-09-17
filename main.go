package main

import (
	hnsw "code.byted.org/zhaoguodong.stan/hnsw/alg"
	"fmt"
	"math/rand"
	"sort"
)

func main() {
	h := hnsw.NewHNSW(100, 8, hnsw.NewL2Space(2))
	var points []*hnsw.Point
	for i := 0; i < 1000; i++ {
		points = append(points, hnsw.NewPoint([]float64{rand.Float64(), rand.Float64()}))
		h.Insert(points[i])
	}
	target := hnsw.NewPoint([]float64{0.5, 0.5})
	knn := h.SearchKNN(target, 20)
	for i, p := range knn {
		fmt.Printf("candidate %d: %v\n", i, p)
	}
	fmt.Println()
	sort.Slice(points, func(i, j int) bool {
		space := hnsw.NewL2Space(2)
		return space.GetDist(target, points[i]) < space.GetDist(target, points[j])
	})
	for i := 0; i < 20; i++ {
		fmt.Printf("nearest %d: (%v, %v)\n", i, hnsw.NewL2Space(2).GetDist(target, points[i]), points[i])
	}
}
