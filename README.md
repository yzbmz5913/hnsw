# hnsw
Hierarchical Navigable Small World, an approximate nearest neighbors Algorithms.
```go
func main() {
	h := hnsw.NewHNSW(100, 8, hnsw.NewL2Space(2))
	var points []*hnsw.Point
	for i := 0; i < 1000; i++ {
		h.Insert(hnsw.NewPoint([]float64{rand.Float64(), rand.Float64()}))
	}
	target := hnsw.NewPoint([]float64{0.5, 0.5})
	knn := h.SearchKNN(target, 20)
	for i, p := range knn {
		fmt.Printf("candidate %d: %v\n", i, p)
	}
}
```
