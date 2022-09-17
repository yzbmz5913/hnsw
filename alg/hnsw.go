package hnsw

import (
	"container/heap"
	"math"
	"math/rand"
)

type HierarchicalNSW struct {
	space Space  // distance function
	topL  int    // top layer level
	ep    *Point // entrypoint
	ef    int    // size of the dynamic candidate list
	m     int    // number of established connections
	m0    int    // max number of connections for each point for layer 0
}

func NewHNSW(ef, m int, space Space) *HierarchicalNSW {
	return &HierarchicalNSW{
		space: space,
		topL:  -1,
		ep:    nil,
		ef:    ef,
		m:     m,
		m0:    m * 2,
	}
}

func (h *HierarchicalNSW) Insert(p *Point) {
	l := int(math.Floor(-math.Log(rand.Float64()) / math.Log(float64(h.m))))

	ep := h.ep
	if ep != nil {
		if l < h.topL {
			curDist := h.space.GetDist(p, ep)
			for level := h.topL; level > l; level-- {
				changed := true
				for changed {
					changed = false
					for _, neighbor := range ep.conn[level] {
						d := h.space.GetDist(p, neighbor)
						if d < curDist {
							curDist = d
							ep = neighbor
							changed = true
						}
					}
				}
			}
		}
		level := l
		if l > h.topL {
			level = h.topL
		}
		for ; level >= 0; level-- {
			cand := h.searchLayer(ep, p, level)
			ep = h.mutuallyConnectTo(p, cand, level)
		}
	} else {
		h.ep = p
		h.topL = l
	}
	if l > h.topL {
		h.ep = p
		h.topL = l
	}
}

func (h *HierarchicalNSW) SearchKNN(q *Point, K int) []*Pair[float64, *Point] {
	ep := h.ep
	for l := h.topL; l >= 1; l-- {
		curDist := h.space.GetDist(q, ep)

		changed := true
		for changed {
			changed = false
			for _, neighbor := range ep.conn[l] {
				d := h.space.GetDist(q, neighbor)
				if d < curDist {
					curDist = d
					ep = neighbor
					changed = true
				}
			}
		}

	}
	cand := h.searchLayer(ep, q, 0)
	i := 0
	ret := make([]*Pair[float64, *Point], 0, K)
	for cand.Len() > 0 && i < K {
		ret = append(ret, (*cand)[0])
		heap.Pop(cand)
		i++
	}
	return ret
}

func (h *HierarchicalNSW) searchLayer(ep *Point, q *Point, lc int) *pQueue {
	visited := make(map[*Point]bool)
	W := &pQueue{} // farthest first
	C := &pQueue{} // nearest first
	visited[ep] = true
	heap.Push(W, NewPair(-h.space.GetDist(q, ep), ep))
	heap.Push(C, NewPair(h.space.GetDist(q, ep), ep))
	for C.Len() > 0 {
		c := (*C)[0]
		f := (*W)[0]
		if c.first > -f.first {
			break
		}
		heap.Pop(C)
		for _, neighbor := range c.second.conn[lc] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			f := (*W)[0]
			d := h.space.GetDist(q, neighbor)
			if d < (-f.first) || W.Len() < h.ef {
				heap.Push(C, NewPair(d, neighbor))
				heap.Push(W, NewPair(-d, neighbor))
				if W.Len() > h.ef {
					heap.Pop(W)
				}
			}
		}
	}
	ret := &pQueue{}
	for W.Len() > 0 {
		heap.Push(ret, NewPair(-(*W)[0].first, (*W)[0].second))
		heap.Pop(W)
	}
	return ret
}

func (h *HierarchicalNSW) mutuallyConnectTo(p *Point, cand *pQueue, level int) *Point {
	mMax := h.m
	if level == 0 {
		mMax = h.m0
	}
	h.getNeighbors(cand, h.m)
	selectedNeighbors := make([]*Point, 0, h.m)
	for cand.Len() > 0 {
		selectedNeighbors = append(selectedNeighbors, (*cand)[0].second)
		heap.Pop(cand)
	}
	ret := selectedNeighbors[len(selectedNeighbors)-1] // the farthest
	// mutually link p and neighbors
	for _, neighbor := range selectedNeighbors {
		p.conn[level] = append(p.conn[level], neighbor)
		if len(neighbor.conn) < mMax {
			neighbor.conn[level] = append(neighbor.conn[level], p)
		} else {
			dMax := h.space.GetDist(p, neighbor)
			cand := &pQueue{}
			heap.Push(cand, NewPair(dMax, p))
			for _, nn := range neighbor.conn[level] {
				heap.Push(cand, NewPair(h.space.GetDist(nn, neighbor), nn))
			}
			h.getNeighbors(cand, mMax)
			neighbor.conn[level] = neighbor.conn[level][:0]
			for cand.Len() > 0 {
				neighbor.conn[level] = append(neighbor.conn[level], (*cand)[0].second)
				heap.Pop(cand)
			}
		}
	}
	return ret
}

func (h *HierarchicalNSW) getNeighbors(cand *pQueue, M int) {
	if cand.Len() < M {
		return
	}
	var ret []*Pair[float64, *Point]
	var closest = &pQueue{} // from farthest to nearest
	for cand.Len() > 0 {
		heap.Push(closest, NewPair(-(*cand)[0].first, (*cand)[0].second))
		heap.Pop(cand)
	}
	for closest.Len() > 0 {
		if len(ret) > M {
			break
		}
		curPair := (*closest)[0]
		dist := -curPair.first
		heap.Pop(closest)
		ok := true
		for _, pair := range ret {
			curDist := h.space.GetDist(pair.second, curPair.second)
			if curDist < dist {
				ok = false
				break
			}
		}
		if ok {
			ret = append(ret, curPair)
		}
	}
	for _, pair := range ret {
		heap.Push(cand, NewPair(-pair.first, pair.second))
	}
}
