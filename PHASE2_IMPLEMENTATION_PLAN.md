# Phase 2: Architectural Improvements - Implementation Plan

## Status: IN PROGRESS

### Quick Win: Remove stdio Duplication (Priority: MEDIUM) - ✅ NEXT
**Estimated time:** 2 hours
**Status:** Starting now

- [x] Analyze stdio_custom.py and stdio_working.py
- [ ] Update client_manager.py imports
- [ ] Delete stdio_custom.py
- [ ] Add deprecation notice if needed
- [ ] Run tests to verify no breakage

### 2.1: Split orchestrator.py (Priority: HIGH)
**Estimated time:** 2 weeks
**Target:** 925 → ~200 lines

#### Week 1
- [ ] Create initializer.py (~150 lines)
- [ ] Create lifecycle.py (~150 lines)
- [ ] Create executor.py (~200 lines)
- [ ] Write unit tests for each module

#### Week 2
- [ ] Refactor main orchestrator.py (~200 lines)
- [ ] Update all imports across codebase
- [ ] Migrate existing tests
- [ ] Verify backward compatibility

### 2.2: Split graph_factory.py (Priority: HIGH)
**Estimated time:** 2 weeks
**Target:** 668 → ~150 lines

#### Week 3
- [ ] Create node_builder.py (~200 lines)
- [ ] Create edge_builder.py (~150 lines)
- [ ] Create graph_validator.py (~150 lines)
- [ ] Write unit tests for each builder

#### Week 4
- [ ] Refactor main graph_factory.py (~150 lines)
- [ ] Update tests
- [ ] Extract common execution wrapper logic
- [ ] Add comprehensive docstrings

## Success Metrics

- [ ] orchestrator.py: 925 → ~200 lines (78% reduction)
- [ ] graph_factory.py: 668 → ~150 lines (78% reduction)
- [ ] stdio duplication: -200 lines
- [ ] All existing tests passing
- [ ] New tests: >80% coverage
- [ ] Backward compatibility maintained
- [ ] Clear separation of concerns
