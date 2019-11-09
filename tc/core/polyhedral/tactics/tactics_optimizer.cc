#include "tc/core/polyhedral/tactics/tactics_optimizer.h"
#include <iostream>
#include "tc/core/polyhedral/matchers/access.h"
#include "tc/core/polyhedral/matchers/access_patterns.h"
#include "tc/core/polyhedral/matchers/matchers.h"
#include "tc/core/polyhedral/matchers/builders.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"

namespace tc {
namespace polyhedral {
namespace tactics {

using namespace matchers;

// Finds all nodes in a schedule tree rooted at root that match m
/*
std::vector<isl::schedule_node> findPatterns(
    const matchers::ScheduleNodeMatcher& m,
    isl::schedule_node root) {
  std::vector<isl::schedule_node> rootMatched;
  std::stack<isl::schedule_node> nodeStack;
  nodeStack.push(root);

  while (nodeStack.empty() == false) {
    root = nodeStack.top();
    nodeStack.pop();

    if (matchers::ScheduleNodeMatcher::isMatching(m, root)) {
      rootMatched.push_back(root);
    }

    size_t n_children =
        static_cast<size_t>(isl_schedule_node_n_children(root.get()));
    for (size_t i = 0; i < n_children; i++) {
      nodeStack.push(root.child(i));
    }
  }

  return rootMatched;
}

// Checks if the node n has already been marked by a matcher
bool hasTacticsMarker(isl::schedule_node n) {
  if (!n.has_parent() || !n.parent().isa<isl::schedule_node_mark>())
    return false;

  isl::schedule_node_mark mark = n.parent().as<isl::schedule_node_mark>();

  return mark.get_id().get_name().rfind("__tactics", 0) == 0;
}

// Checks if the node n or any of its ancestors up to the root have
// already been marked by a matcher
bool selfOrAncestorHasTacticsMarker(isl::schedule_node n) {
  if (hasTacticsMarker(n))
    return true;

  if (n.has_parent())
    return selfOrAncestorHasTacticsMarker(n.parent());
  else
    return false;
}

/// Simple wrapper around isl::ctx that allocates a new context during
/// construction and frees it during destruction, e.g. when a stack-allocated
/// instance of ScopedCtx goes out of scope.
/// Implicitly convertible to both isl_ctx* and isl::ctx for convenience.
/// Intentionally not copy-constructible or copy-assignable as it would have
/// required reference counting.  Move-constructible to enable ownership
/// transfer.
class ScopedCtx {
 public:
  ScopedCtx() : ctx(isl_ctx_alloc()) {}
  explicit ScopedCtx(isl::ctx&& ctx) : ctx(ctx) {}
  ScopedCtx(const ScopedCtx&) = delete;
  ScopedCtx(ScopedCtx&&) = default;
  ~ScopedCtx() {
    isl_ctx_free(ctx.release());
  }

  ScopedCtx& operator=(const ScopedCtx&) = delete;
  ScopedCtx& operator=(ScopedCtx&&) = default;

  operator isl::ctx() {
    return ctx;
  }
  operator isl_ctx*() {
    return ctx.get();
  }

 private:
  isl::ctx ctx;
};

bool operator==(
    const Halide::Internal::Variable& a,
    const Halide::Internal::Variable& b) {
  return a.name == b.name;
}

bool operator!=(
    const Halide::Internal::Variable& a,
    const Halide::Internal::Variable& b) {
  return !(a == b);
}

// Collects information about a tensor indexed by a set of variables,
// one variable per dimension
class TensorAccess {
 public:
  std::string tensor;
  std::vector<const Halide::Internal::Variable*> dims;

  bool operator==(const TensorAccess& b) const {
    return tensor == b.tensor &&
        std::equal(
               dims.begin(),
               dims.end(),
               b.dims.begin(),
               [](const Halide::Internal::Variable* const va,
                  const Halide::Internal::Variable* const vb) -> bool {
                 return *va == *vb;
               });
  }

  bool operator!=(const TensorAccess& b) const {
    return !(*this == b);
  }
};

std::ostream& operator<<(std::ostream& os,
    const Halide::Internal::Variable& v) {
  os << v.name;
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorAccess& ta) {
  os << ta.tensor << "(" << (*ta.dims[0]) << ", " << (*ta.dims[1]) << ")";
  return os;
}

// Extracts the minimum and maximum values for a given dimension from
// a set of statement instances. If the dimension is not contiguous or
// does not have contant bounds, the function returns
// false. Otherwise, the bounds are provided in min and max and the
// function returns true.
bool extractDimBounds(
    isl::set set,
    const std::string& dimName,
    int& min,
    int& max) {
  isl::space space = set.get_space();

  int dimIdx = space.find_dim_by_name(isl::dim::out, dimName);

  if (dimIdx == -1)
    return false;

  // Extract constants for bounds from piecewise quasi-affine
  // expressions for the minimum and maximum
  isl::pw_aff minAff = set.dim_min(dimIdx);
  isl::pw_aff maxAff = set.dim_max(dimIdx);

  // Check that the expressions are all constant and constructed of a
  // single piece
  if (!minAff.is_cst() || !maxAff.is_cst() || minAff.n_piece() != 1 ||
      maxAff.n_piece() != 1) {
    return false;
  }

  minAff.foreach_piece([&](isl::set s, isl::aff a) {
    min = std::stoi(a.get_constant_val().to_str());
  });

  maxAff.foreach_piece([&](isl::set s, isl::aff a) {
    max = std::stoi(a.get_constant_val().to_str()) + 1;
  });

  return true;
}

// Optimizer that marks matrix multiplications of the form
//
//   C = alpha * A * B
//
// where alpha is a constant. Associativity and commutativity is
// supported for the multiplication, i.e., the optimizer matches
//
//   C = (alpha * A) * B
//   C = alpha * (A * B)
//   C =  (A * alpha) * B
//   C =  A * (alpha * B)
//   etc.
class GemmOptimizer {
 public:
  // Returns the tactics matcher for the schedule tree. The matcher
  // only checks the shape of the subtree, but verifies neither
  // accesses nor operations
  static ScheduleNodeMatcher getMatcher() {


    return band(band(sequence(filter(leaf()), filter(leaf()))));
  }

  // Extracts the components of a n-D tensor access from an expression
  // (e.g., C(i, j)) if expr is an n-D tensor access. Otherwise the
  // function returns false.
  template <std::size_t n>
  static bool extractTensorAccessND(
      const Halide::Expr& expr,
      TensorAccess& ret) {
    const Halide::Internal::Call* call = expr.as<Halide::Internal::Call>();

    if (!call || call->args.size() != n)
      return false;

    std::vector<const Halide::Internal::Variable*> vars;

    for (auto& arg : call->args) {
      auto v = arg.as<Halide::Internal::Variable>();

      if (!v)
        return false;

      vars.push_back(v);
    }

    ret.tensor = call->name;
    ret.dims = vars;

    return true;
  }

  // Extracts the name of a tensor of a 0-dimensional tensor access
  // from an expression (e.g., alpha()) if expr is a 0D tensor
  // access. Otherwise the function returns false.
  static bool extractTensorAccess0D(
      const Halide::Expr& expr,
      TensorAccess& ret) {
    return extractTensorAccessND<0>(expr, ret);
  }

  // Extracts the components of a 2D tensor access from an expression
  // (e.g., C(i, j)) if expr is a 2D tensor access. Otherwise the
  // function returns false.
  static bool extractTensorAccess2D(
      const Halide::Expr& expr,
      TensorAccess& ret) {
    return extractTensorAccessND<2>(expr, ret);
  }

  // Extracts all operands of a multiplication of the form (a * (b *
  // (c * (...))) in order from expr.
  static void extractMultiplicationOperands(
      const Halide::Expr& expr,
      std::vector<const Halide::Expr*>& operands) {
    const Halide::Internal::Mul* mul;

    if ((mul = expr.as<Halide::Internal::Mul>())) {
      extractMultiplicationOperands(mul->a, operands);
      extractMultiplicationOperands(mul->b, operands);
    } else {
      operands.push_back(&expr);
    }
  }

  // Extracts the operands of the matrix multiplication from the root
  // of the match obtained from the matcher. On success, the operands
  // are provided in mmi and the function return true. If the
  // operations and accesses in the subtree do not match a matrix
  // multiplication, the function returns false.
  static bool extractMatMulInfo(
      isl::schedule_node m,
      const MappedScop& scop,
      MatMulInfo& mmi) {
    auto& lscop = scop.scop();

    // The ID of the leaf node of the init statement maps to the halide
    // statement
    isl::schedule_node initLeaf = m.child(0).child(0).child(0).child(0);
    const Halide::Internal::Stmt& initStatement =
        getHalideStatement(lscop, initLeaf);

    // The init statement should be of the form
    //
    //   C(m, k) = 0.000000f
    const Halide::Internal::Provide* initProvide =
        initStatement.as<Halide::Internal::Provide>();

    if (!initProvide || initProvide->args.size() != 2)
      return false;

    TensorAccess accC_init;
    accC_init.tensor = initProvide->name;
    accC_init.dims = {initProvide->args[0].as<Halide::Internal::Variable>(),
                      initProvide->args[1].as<Halide::Internal::Variable>()};

    int minM_init, maxM_init;
    int minK_init, maxK_init;

    isl::set initSet = initLeaf.get_domain().get_set_list().get_at(0);

    if (!extractDimBounds(
            initSet, accC_init.dims[0]->name, minM_init, maxM_init) ||
        !extractDimBounds(
            initSet, accC_init.dims[1]->name, minK_init, maxK_init)) {
      return false;
    }

    if (minM_init != 0 || minK_init != 0)
      return false;

    // RHS of initilization must be a single constant
    if (initProvide->values.size() != 1)
      return false;

    const Halide::Internal::FloatImm* initVal =
        initProvide->values[0].as<Halide::Internal::FloatImm>();

    if (!initVal)
      return false;

    if (initVal->value != 0.0)
      return false;

    // The ID of the leaf node of the core statement maps to the halide
    // statement
    isl::schedule_node coreLeaf = m.child(0).child(0).child(1).child(0);
    const Halide::Internal::Stmt& coreStatement =
        getHalideStatement(lscop, coreLeaf);

    // Core statement should be a provide statement for a 2D tensor of
    // the form
    //
    //   C(m, k) = ReductionUpdate((C(m, k) + ((alpha() * A(m, n)) * B(n, k))))
    //
    const Halide::Internal::Provide* coreProvide =
        coreStatement.as<Halide::Internal::Provide>();

    if (!coreProvide || coreProvide->args.size() != 2)
      return false;

    // Extract 2D tensor access from provide node
    TensorAccess accC_LHS;
    accC_LHS.tensor = coreProvide->name;
    accC_LHS.dims = {coreProvide->args[0].as<Halide::Internal::Variable>(),
                     coreProvide->args[1].as<Halide::Internal::Variable>()};

    if (!accC_LHS.dims[0] || !accC_LHS.dims[1])
      return false;

    // RHS must be a single call to ReductionUpdate
    if (coreProvide->values.size() != 1)
      return false;

    const Halide::Internal::Call* reductionCall =
        coreProvide->values[0].as<Halide::Internal::Call>();

    if (!reductionCall || !reductionCall->is_intrinsic("ReductionUpdate") ||
        reductionCall->args.size() != 1) {
      return false;
    }

    // Check that the reduction is an addition
    const Halide::Internal::Add* addition =
        reductionCall->args[0].as<Halide::Internal::Add>();

    if (!addition)
      return false;

    // Determine if output matrix is the first or second operand of
    // the addition, i.e., if
    //
    //   C(m, k) = ReductionUpdate((C(m, k) + ((alpha() * A(m, n)) * B(n, k))))
    //
    // or
    //
    //  C(m, k) = ReductionUpdate((((alpha() * A(m, n)) * B(n, k))) + C(m, k))
    //
    TensorAccess accC_RHS;
    const Halide::Expr* multiplication;

    if (extractTensorAccess2D(addition->a, accC_RHS)) {
      multiplication = &addition->b;
    } else if (extractTensorAccess2D(addition->b, accC_RHS)) {
      multiplication = &addition->a;
    } else {
      return false;
    }

    // Check that matrix access on the LHS and RHS reference same
    // matrix and same dimensions
    if (accC_LHS != accC_RHS)
      return false;

    // Extract input matrices A and B from multiplication
    std::vector<const Halide::Expr*> mulOperands;

    extractMultiplicationOperands(*multiplication, mulOperands);

    if (mulOperands.size() != 3)
      return false;

    // Operands must be two 2D tensor accesses and a constant
    std::vector<TensorAccess> accAB;
    TensorAccess accAlpha;
    bool hasAlpha = false;

    for (const Halide::Expr* mulOperand : mulOperands) {
      TensorAccess tmp;

      if (extractTensorAccess2D(*mulOperand, tmp))
        accAB.push_back(tmp);
      else if (extractTensorAccess0D(*mulOperand, tmp)) {
        hasAlpha = true;
        accAlpha = tmp;
      }
    }

    if (accAB.size() != 2 || !hasAlpha)
      return false;

    // Output Matrix C must not appear again in the multiplication
    if (accAB[0].tensor == accC_LHS.tensor ||
        accAB[1].tensor == accC_LHS.tensor) {
      return false;
    }

    // The first dimension A must match the first dimension of the
    // output matrix, the first dimension B must match the second
    // dimensions of A and the second dimension of A must match the
    // first dimension of B.
    const TensorAccess* accA = nullptr;
    const TensorAccess* accB = nullptr;

    // First, determine order of A and B based on matching of their
    // dimensions
    if (*accAB[0].dims[0] == *accAB[1].dims[1]) {
      accA = &accAB[1];
      accB = &accAB[0];
    } else if (*accAB[1].dims[0] == *accAB[0].dims[1]) {
      accA = &accAB[0];
      accB = &accAB[1];
    } else {
      return false;
    }

    // Then, check if the dimensions correspond to those of the output
    // matrix C
    if (*accA->dims[0] != *accC_LHS.dims[0] ||
        *accB->dims[1] != *accC_LHS.dims[1]) {
      return false;
    }

    // Extract the extents for A, B and C from the polyhedral
    // representation. The variable names used to index the tensors
    // are the same as the dimensions for the statement
    // instances. Simply extract M, N and K from the set of statement
    // instances of the form:
    //
    //    S[i, j, k] ... 0 <= i <= M and 0 <= j <= N and 0 <= k <= K
    //
    int minM, minN, minK;
    int maxM, maxN, maxK;

    isl::set coreSet = coreLeaf.get_domain().get_set_list().get_at(0);

    if (!extractDimBounds(coreSet, accA->dims[0]->name, minM, maxM) ||
        !extractDimBounds(coreSet, accA->dims[1]->name, minN, maxN) ||
        !extractDimBounds(coreSet, accB->dims[1]->name, minK, maxK)) {
      return false;
    }

    if (minM != 0 || minN != 0 || minK != 0)
      return false;

    // The access to the output matrix of the core statement must
    // match the one of the initilization statement
    if (accC_LHS.tensor != accC_init.tensor || maxM != maxM_init ||
        maxK != maxK_init) {
      return false;
    }

    // Set the actual tensor names
    mmi.C = accC_LHS.tensor;
    mmi.A = accA->tensor;
    mmi.B = accB->tensor;
    mmi.alpha = accAlpha.tensor;
    mmi.m = maxM;
    mmi.n = maxN;
    mmi.k = maxK;

    return true;
  }

  // Check if a matched subtree has the right operations
  static bool checkMatch(isl::schedule_node m, const MappedScop& scop) {
    MatMulInfo mmi;

    return extractMatMulInfo(m, scop, mmi);
  }

  // Processes a match: Marks the root node of the match as a matrix
  // multiplication and adds the matrix multiplication information to
  // the replacements to be performed upon code generation.
  static isl::schedule_node processMatch(
      isl::schedule_node m,
      const MappedScop& scop,
      TacticsReplacements& replacements) {
    isl::id markId = isl::id::alloc(m.get_ctx(), "__tactics_gemm", nullptr);
    MatMulInfo mmi;

    if (extractMatMulInfo(m, scop, mmi))
      replacements.matmul.emplace(std::make_pair(markId, mmi));

    return m.insert_mark(markId);
  }
};

isl::schedule optimizeGemmSchedule(
    const MappedScop& scop,
    TacticsReplacements& replacements) {
  isl::schedule schedule = toIslSchedule(scop.schedule());
  isl::schedule_node root = schedule.get_root();

  auto matcherGEMM = GemmOptimizer::getMatcher();

  bool restart;

  do {
    restart = false;
    std::vector<isl::schedule_node> matches = findPatterns(matcherGEMM, root);

    for (auto& m : matches) {
      // Make sure that the same subtree is not matched twice and that
      // matches do not overlap
      if (selfOrAncestorHasTacticsMarker(m))
        continue;

      if (GemmOptimizer::checkMatch(m, scop)) {
        root = GemmOptimizer::processMatch(m, scop, replacements);

        // When processing a match previous nodes might get
        // invalidated -> restart matching
        restart = true;
        break;
      }
    }
  } while (restart);

  schedule = root.get_schedule();

  return schedule;
}
*/

enum class GemmType {isNT, isNN, isTN};
enum class GemvType {isNT, isNN};
constexpr int TILE_FACTOR_CIM_DEVICE = 512;

// utility function.
//
// @param node: Current node where to start cutting.
// @param replacement: Subtree to be attached after @p node.
// @return: Root node of the rebuild subtree.
//
// NOTE: This is not always possible. Cutting children
// in set or sequence is not allowed by ISL and as a consequence
// by Loop Tactics.
static isl::schedule_node
rebuild(isl::schedule_node node,
        const builders::ScheduleNodeBuilder &replacement) {

  node = node.cut();
  node = replacement.insertAt(node);
  return node;
}

// check schedule dimensionality
static bool scheduleDimensionality(isl::union_map schedule, 
  int dimensionality) {

  if (schedule.n_map() != 1)
    return false;
  isl::map scheduleAsMap = isl::map::from(schedule);
  int dims = scheduleAsMap.dim(isl::dim::in);
  if (dims != dimensionality)
    return false;
  return true;
}

// Returns the Halide statement associated to a schedule node
// n. Throws if no Halide statement is associated with the node.
static const Halide::Internal::Stmt& getHalideStatement(
    const Scop& scop,
    isl::schedule_node n) {
  auto dom = n.get_domain();
  isl::set_list instances = dom.get_set_list();
  isl::set instance = instances.get_at(0);
  isl::id instId = instance.get_tuple_id();

  return scop.halide.statements.at(instId);
}

// wrap sub-tree in case of match. If "node" matches with
// "pattern" then wrap node and it's sub-tree with a mark node
// named "marker".
template <typename T>
isl::schedule_node wrapNodeIfMatches(
    isl::schedule_node node,
    const matchers::ScheduleNodeMatcher& pattern,
    std::string& marker,
    T& p) {
  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    if (marker == "__tactics_mvt")
      node = node.insert_mark(
          isl::id::alloc(node.get_ctx(), marker, new GemvInfo{p.mvi}));
    else if (marker == "__tactics_gemm")
      node = node.insert_mark(
          isl::id::alloc(node.get_ctx(), marker, new MatMulInfo{p.mmi}));
    else if (marker == "__tactics_batched_gemm")
      node = node.insert_mark(
        isl::id::alloc(node.get_ctx(), marker, new BatchedMatMulInfo{p.bmmi}));
    else if (marker == "__tactics_gemm_is_dominated")
      node = node.insert_mark(
        isl::id::alloc(node.get_ctx(), marker, new MatMulInfo{p.mmi}));
    else {
      std::cout << marker << std::endl; 
      assert(0 && "case not defined");
    }
  }

  return node;
}

// walk schedule tree starting from "node". If "node"
// matches with "pattern" wrap the sub-tree with a
// mark node named "marker".
template <typename T>
isl::schedule_node wrapOnMatch(
    isl::schedule_node node,
    const matchers::ScheduleNodeMatcher& pattern,
    std::string& marker,
    T& p) {
  node = wrapNodeIfMatches(node, pattern, marker, p);

  // break recursion
  if (node.isa<isl::schedule_node_mark>())
    return node;

  for (int i = 0; i < node.n_children(); i++) {
    node = wrapOnMatch(node.child(i), pattern, marker, p).parent();
  }
  return node;
}

isl::schedule_node rebuildIfMatches(isl::schedule_node node,
    const matchers::ScheduleNodeMatcher& pattern,
    const builders::ScheduleNodeBuilder& replacement) {
  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node))
    node = rebuild(node, replacement);
  return node;
}

isl::schedule_node rebuildOnMatch(
    isl::schedule_node node,
    const matchers::ScheduleNodeMatcher& pattern, 
    const builders::ScheduleNodeBuilder& replacement) {
  node = rebuildIfMatches(node, pattern, replacement);
  
  if (node.isa<isl::schedule_node_mark>())
    return node;
  
  for (int i = 0; i < node.n_children(); i++)
    node = rebuildOnMatch(node.child(i), pattern, replacement).parent();

  return node;
}

// get scheduled read and write accesses restricted to "node".
std::pair<isl::union_map, isl::union_map> getReadsAndWritesForNode(
    const MappedScop& scop,
    isl::schedule_node node) {
  auto& lscop = scop.scop();
  auto schedule = node.get_prefix_schedule_union_map();
  schedule = schedule.intersect_domain(node.get_domain());
  auto reads = lscop.body.reads.curry().apply_domain(schedule);
  auto writes = lscop.body.writes.curry().apply_domain(schedule);
  return std::make_pair(reads, writes);
}

// make sure we are dealing with a reduction with + and *
static bool isReduction(const Halide::Internal::Stmt& stmt) {
  const Halide::Internal::Provide* coreProvide =
      stmt.as<Halide::Internal::Provide>();
  if (!coreProvide)
    return false;

  // check reduction.
  const Halide::Internal::Call* reductionCall =
      coreProvide->values[0].as<Halide::Internal::Call>();

  if ((!reductionCall) || (!reductionCall->is_intrinsic("ReductionUpdate")) ||
      (reductionCall->args.size() != 1))
    return false;

  // check addition for reduction.
  const Halide::Internal::Add* add =
      reductionCall->args[0].as<Halide::Internal::Add>();
  if (!add)
    return false;

  // FIXME: recursively walk the mult and make sure
  // all the nested operations are also mult.
  if (!(add->a.as<Halide::Internal::Mul>()) &&
      !(add->b.as<Halide::Internal::Mul>()))
    return false;

  return true;
}

// check operations for GEMV
static bool checkOperationsInGemvCore(
    const MappedScop& scop,
    isl::schedule_node leaf) {
  auto& lscop = scop.scop();
  const Halide::Internal::Stmt& stmt = getHalideStatement(lscop, leaf);
  return isReduction(stmt);
}

// check operations for GEMM
static bool checkOperationsInGemmCore(
    const MappedScop& scop,
    isl::schedule_node leaf) {
  auto& lscop = scop.scop();
  const Halide::Internal::Stmt& stmt = getHalideStatement(lscop, leaf);
  return isReduction(stmt);
}

// check operations in a generic initialization statement.
// RHS single constant set to 0.0f.
static bool checkOperationsInInit(
    const MappedScop& scop,
    isl::schedule_node leaf) {
  auto& lscop = scop.scop();

  const Halide::Internal::Stmt& stmt = getHalideStatement(lscop, leaf);
  const Halide::Internal::Provide* initProvide =
      stmt.as<Halide::Internal::Provide>();

  if ((!initProvide)) {
    return false;
  }

  // RHS of initialization must be a single constant.
  if (initProvide->values.size() != 1) {
    return false;
  }

  const Halide::Internal::FloatImm* initVal =
      initProvide->values[0].as<Halide::Internal::FloatImm>();

  if ((!initVal) || (initVal->value != 0.0)) {
    return false;
  }
  return true;
}

// get the array name from "map"
static std::string getName(isl::map map) {
  if (map.can_uncurry())
    map = map.uncurry();

  isl::set set = map.range();

  if (set.has_tuple_id())
    return set.get_tuple_id().to_str();

  std::cout << map.to_str() << "\n";
  assert(0 && "Cannot get array name from map");
  return "nullptr";
}

template <class T, class...>
struct are_same : std::true_type {};

template <class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<
          bool,
          std::is_same<T, U>{} && are_same<T, TT...>{}> {};

// check if "a" and "b" have the same elements.
static bool isEqual(std::vector<int>& a, std::vector<int>& b) {
  if (a.size() != b.size())
    return false;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  if (std::equal(a.begin(), a.begin() + a.size(), b.begin()))
    return true;
  return false;
}

// given map "map" and a set of matched dimension "args"
// return true if the map dimension match with "args".
// we map is scheduled. FIXME: We can avoid this by capturing
// array name directly in arrayPlaceholder(...)
template <typename... Args>
static bool matchDimImpl(isl::map map, Args... args) {
  std::cout << __func__ << std::endl;
  std::vector<int> matchedDims{args...};
  std::vector<int> mapsDims{};

  isl::pw_multi_aff muaff = isl::pw_multi_aff::from_map(map);
  for (size_t i = 0; i < map.dim(isl::dim::out); i++) {
    isl::pw_aff pwaff = muaff.get_pw_aff(i);
    pwaff.foreach_piece([&](isl::set set, isl::aff aff) -> isl_stat {
      for (size_t j = 0; j < map.dim(isl::dim::in); j++) {
        isl::val val = aff.get_coefficient_val(isl::dim::in, j);
        if (!val.is_zero())
          mapsDims.push_back(j);
      }
      return isl_stat_ok;
    });
  }

  return isEqual(matchedDims, mapsDims);
}

// Among the maps in "maps" return the only map which has dimensions
// that match with "args". "args" is a set of dimension matched with
// the access relation matchers. This function makes sense if we always
// expect a single match.
template <typename... Args>
static isl::map matchDim(const std::vector<isl::map>& maps, Args... args) {
  for (const auto& m : maps)
    if (matchDimImpl(m, args...))
      return m;

  for (const auto& m : maps)
    std::cout << m.to_str() << std::endl;
  std::vector<int> matchedDims{args...};
  for (const auto i : matchedDims)
    std::cout << "matched dim is : " << i << std::endl;
  assert(0);
  return maps[0];
}

// given the union map "umap" return a std::vector of map
static std::vector<isl::map> vectorMapsFromUnionMap(isl::union_map umap) {
  std::vector<isl::map> tmp{};
  umap.foreach_map([&](isl::map m) -> isl_stat {
    tmp.push_back(m);
    return isl_stat_ok;
  });

  return tmp;
}

std::vector<DimInfo>extractDimInfoFromRange(isl::set range) {

  std::vector<DimInfo>res{};
  if (range.n_dim() == 0)
    return res;
  
  size_t dims = range.n_dim();
  for (size_t i = 0; i < dims; i++) {
    
    auto min = range.dim_min(i);
    auto max = range.dim_max(i);
    
    assert(min.n_piece() == 1);
    assert(max.n_piece() == 1);
  
    isl::val minVal;
    isl::val maxVal;
  
    min.foreach_piece([&](isl::set s, isl::aff aff) {
      minVal = aff.get_constant_val();
      return isl_stat_ok;
    });
    max.foreach_piece([&](isl::set, isl::aff aff) {
      maxVal = aff.get_constant_val();
      return isl_stat_ok;
    });

    res.push_back(DimInfo{std::stoi(minVal.to_str()),
                          (std::stoi(maxVal.to_str()) + 1)});
  }

  return res;
}

// return the array name of the map among "umap" which has dimensions
// that matches with "args". "args" is a set of dimensions matched with
// the access relation matchers.
template <typename... Args>
static ArrayInfo getArrayNameAndExtensionFromMap(isl::union_map umap, Args... args) {
  static_assert(are_same<int, Args...>{}, "must be of type int");

  std::vector<isl::map> maps = vectorMapsFromUnionMap(umap);

  isl::map result = matchDim(maps, args...);
  assert(!result.is_null() && "expect not null");
  auto name = getName(result);
  auto range = result.range(); 
  return ArrayInfo{name, extractDimInfoFromRange(range)};
}

// return the map among "maps" which has dimensions that
// matches with "args"
template <typename... Args>
static isl::map getMapFromUnionMap(isl::union_map umap, Args... args) {
  static_assert(are_same<int, Args...>{}, "must be of type int");

  std::vector<isl::map> maps = vectorMapsFromUnionMap(umap);
  
  isl::map result = matchDim(maps, args...);
  return result;
}

// return true if the access has dimensionality x
std::function<bool(isl::map map)> isXd(size_t x) {
  return [x](isl::map map) { 
    if (map.dim(isl::dim::out) != x)  
      return false;
    return true;
  };
}

std::function<bool(isl::map map)> isConstant() {

  return [](isl::map map) {
    if (map.dim(isl::dim::out) != 0)
      return false;
    return true;
  };
}

// _allOf as in
// https://www.boost.org/doc/libs/1_52_0/libs/algorithm/doc/html/algorithm/CXX11.html
static bool _allOf(isl::union_map umap, std::function<bool(isl::map)> f) {
  std::cout << __func__ << std::endl;
  auto unionMapAsMaps = vectorMapsFromUnionMap(umap);
  for (const auto& m : unionMapAsMaps)
    if (!f(m))
      return false;
  return true;
}

// _oneOf
static bool _oneOf(isl::union_map umap, std::function<bool(isl::map)> f) {
  std::cout << __func__ << std::endl;
  auto unionMapAsMaps = vectorMapsFromUnionMap(umap);
  auto counter = 0;
  for (const auto& m : unionMapAsMaps)
    if (f(m))
      counter++;
  return counter == 1;
}

// Create a map in the provided domain which maps from one
// element of the provided domain to another element. The mapping 
// is made such that all point in the map are equal exept for
// the dimension indexes by posDim. The dimension indexes by 
// posDim will be such that the input dimension is strictly smaller
// than the output one.
static isl::map getEqualAndLarger(int posDim, isl::space dom) {
  isl::space space = dom.map_from_set();
  isl::map map = isl::map::universe(space);
  int lastDim = map.dim(isl::dim::in);
  for (int i = 0; i < lastDim; i++) {
    if (i == posDim)
      continue;
    map = map.equate(isl::dim::in, i, isl::dim::out, i);
  }
  map = map.order_lt(isl::dim::in, posDim, isl::dim::out, posDim);
  return map;
}

// get the stride on dimension "dim"
static int getStrideOnDim(isl::map schedule, int dim, isl::map access) {

  isl::space space = schedule.get_space().range();
  isl::map nextScatt = getEqualAndLarger(dim, space);
  nextScatt = nextScatt.apply_domain(access);
  nextScatt = nextScatt.apply_range(access);
  nextScatt = nextScatt.lexmin();
  isl::set deltas = nextScatt.deltas();

  isl::pw_aff aff = deltas.dim_max(0); 
  assert(aff.n_piece() == 1 && "expect single piece");
  int stride = -1;
  aff.foreach_piece([&](isl::set s, isl::aff a) {
    stride = std::stoi(a.get_constant_val().to_str());
  });

  return stride;
}

// check access pattern for initialization statement in GEMV
// We expect y(i) = 0.0f
//
static bool hasGemvInitPatternImpl(
    const MappedScop& scop,
    isl::schedule_node leaf,
    GemvInfo& mvi) {
  std::cout << __func__ << std::endl;
  auto schedule = leaf.get_prefix_schedule_union_map();
  if (!scheduleDimensionality(schedule, 1))
    return false;
  auto readsAndWrites = getReadsAndWritesForNode(scop, leaf);
  auto reads = readsAndWrites.first;
  auto writes = readsAndWrites.second;

  // we assume initialization statement with zero.
  // so we expect only a write.
  if ((reads.n_map() != 0) && (writes.n_map() != 1))
    return false;

  if (!_allOf(writes, isXd(1)))
    return false;

  isl::ctx ctx = leaf.get_ctx();
  using namespace matchers;
  auto _i = placeholder(ctx);
  auto writeMatches = match(writes, allOf(access(_i)));

  if (writeMatches.size() != 1)
    return false;

  bool operation = false;
  try {
    operation = checkOperationsInInit(scop, leaf);
  } catch (...) {
    return false;
  }

  if (!operation)
    return false;

  auto iPos = writeMatches[0][_i].payload().inputDimPos_;

  mvi.writeToY = getArrayNameAndExtensionFromMap(writes, iPos); 
  mvi.i = writeMatches[0][_i].payload().inputDimPos_;
  isl::map m = getMapFromUnionMap(writes, iPos);
  // TODO: this may be not needed as the access pattern matchers
  // will not allow any increment for the induction. Check if TC
  // allows strided loops. Same for incx.
  mvi.incy = getStrideOnDim(isl::map::from(schedule), iPos, m);
  return true;
}

std::tuple<bool, int, int, GemvType> lookUpPermutationOnReadGemv(isl::ctx ctx,
    isl::union_map reads, int iPos) {
  using namespace matchers;
  auto _i = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto _A = arrayPlaceholder();
  auto _x = arrayPlaceholder();
  auto _y = arrayPlaceholder();
  auto psRead = 
    allOf(access(_A, _i, _j), access(_y, _j), access(_x, _i));
  auto readMatches = match(reads, psRead);
  
  if (readMatches.size() != 1)
    return std::make_tuple(false, 0, 0, GemvType::isNN);
  
  auto iPosNN_NT = readMatches[0][_i].payload().inputDimPos_;
  auto jPosNN_NT = readMatches[0][_j].payload().inputDimPos_;
 
  if (iPos == iPosNN_NT)
    return std::make_tuple(true, iPos, jPosNN_NT, GemvType::isNN);
  
  if (iPos == jPosNN_NT)
    return std::make_tuple(true, iPos, iPosNN_NT, GemvType::isNT);

  return std::make_tuple(false, 0, 0, GemvType::isNN);
}

std::tuple<bool, int, int, GemvType> isGemvCorePattern(isl::ctx ctx,
    isl::union_map reads, isl::union_map writes) {
  using namespace matchers;
  auto _i = placeholder(ctx);
  auto psWrite = allOf(access(_i));
  auto writeMatches = match(writes, psWrite);
  
  if (writeMatches.size() != 1) {
    return std::make_tuple(false, 0, 0, GemvType::isNN);
  }

  auto iPos = writeMatches[0][_i].payload().inputDimPos_;
  auto matchedDims = lookUpPermutationOnReadGemv(ctx, reads, iPos);
  if (!std::get<0>(matchedDims)) {
    return std::make_tuple(false, 0, 0, GemvType::isNN);
  }

  iPos = std::get<1>(matchedDims);
  auto jPos = std::get<2>(matchedDims);
  auto type = std::get<3>(matchedDims);

  return std::make_tuple(true, iPos, jPos, type);
}

// check access pattern for core statement in GEMV
// We expect y(i) = y(i) + x(j) * A[i][j] _OR_
//  y(i) + x(j) * A[j][i]
//
static bool hasGemvCorePatternImpl(
    const MappedScop& scop,
    isl::schedule_node leaf,
    GemvInfo& mvi) {
  std::cout << __func__ << std::endl;
  auto schedule = leaf.get_prefix_schedule_union_map();
  if (!scheduleDimensionality(schedule, 2))
    return false;
  auto readsAndWrites = getReadsAndWritesForNode(scop, leaf);
  auto reads = readsAndWrites.first;
  auto writes = readsAndWrites.second;

  if ((reads.n_map() != 3) && (writes.n_map() != 1))
    return false;

  // TODO: check accesses dimensionality.
  auto matches = isGemvCorePattern(leaf.get_ctx(), reads, writes);
  if (!std::get<0>(matches))
    return false;
  auto iPos = std::get<1>(matches);
  auto jPos = std::get<2>(matches);
  auto type = std::get<3>(matches);

  bool operation = false;
  try {
    operation = checkOperationsInGemvCore(scop, leaf);
  } catch (...) {
    return false;
  }

  if (!operation)
    return false;

  mvi.j = jPos;
  mvi.readFromA = getArrayNameAndExtensionFromMap(
      reads, iPos, jPos);
  mvi.readFromY =
      getArrayNameAndExtensionFromMap(reads, iPos);
  mvi.readFromX =
      getArrayNameAndExtensionFromMap(reads, jPos);
  mvi.writeToY = 
    getArrayNameAndExtensionFromMap(writes, iPos);
  mvi.isAtranspose = (type == GemvType::isNT) ? true : false;
  auto matchedMap = getMapFromUnionMap(reads, jPos);
  mvi.incx = getStrideOnDim(isl::map::from(schedule), jPos, matchedMap); 
  return true;
}

// collect access name for a constant access in "umap"
// We assume only one constant access in "umap" and 
// we expect to find it.
isl::map collectConstant(isl::union_map umap) {

  auto unionMapAsMaps = vectorMapsFromUnionMap(umap);
  for (const auto& m : unionMapAsMaps) 
    if (m.dim(isl::dim::out) == 0)
      return m;
  assert(0 && "cannot find any constant");
  return unionMapAsMaps[0];
}

// collect access name for a constant map in "umap"
std::string collectConstantName(isl::union_map umap) {

  auto tmp = collectConstant(umap);
  return getName(tmp);
}

// is the access pattern C(i,j) = 0 in "writes"?
static std::tuple<bool, int, int> isGemmInitPattern(isl::ctx ctx,
  isl::union_map writes) {
  using namespace matchers;
  auto _i = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto matches = match(writes, allOf(access(_i, _j)));

  if (matches.size() != 1)
    return std::make_tuple(false, 0, 0);

  auto iPos = matches[0][_i].payload().inputDimPos_;
  auto jPos = matches[0][_j].payload().inputDimPos_; 
  return std::make_tuple(true, iPos, jPos);
}

std::tuple<bool, int, int, int, GemmType> lookUpPermutationOnReadGemm(isl::ctx ctx,
  isl::union_map reads, int iPos, int jPos) {
  using namespace matchers; 
  auto _i = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto _k = placeholder(ctx);
  auto _A = arrayPlaceholder();
  auto _B = arrayPlaceholder();
  auto _C = arrayPlaceholder();
  auto psReadNN =
      allOf(access(_A, _i, _j), access(_B, _i, _k), access(_C, _k, _j));
  auto psReadNT =
      allOf(access(_A, _i, _j), access(_B, _i, _k), access(_C, _j, _k));
  auto psReadTN =
      allOf(access(_A, _i, _j), access(_B, _k, _i), access(_C, _k, _j));
  auto readMatchesNN = match(reads, psReadNN);
  auto readMatchesNT = match(reads, psReadNT); 
  auto readMatchesTN = match(reads, psReadTN);

  auto iPosNN = readMatchesNN[0][_i].payload().inputDimPos_;
  auto jPosNN = readMatchesNN[0][_j].payload().inputDimPos_;
  auto kPosNN = readMatchesNN[0][_k].payload().inputDimPos_;
  
  if ((readMatchesNN.size() == 1) && (iPos == iPosNN) && (jPos == jPosNN))
    return std::make_tuple(true, iPosNN, jPosNN, kPosNN, GemmType::isNN);

  auto iPosNT = readMatchesNT[0][_i].payload().inputDimPos_;
  auto jPosNT = readMatchesNT[0][_j].payload().inputDimPos_;
  auto kPosNT = readMatchesNT[0][_k].payload().inputDimPos_;

  if ((readMatchesNT.size() == 1) && (iPos == iPosNT) && (jPos == jPosNT))
    return std::make_tuple(true, iPosNT, jPosNT, kPosNT, GemmType::isNT);

  auto iPosTN = readMatchesTN[0][_i].payload().inputDimPos_;  
  auto jPosTN = readMatchesTN[0][_j].payload().inputDimPos_;
  auto kPosTN = readMatchesTN[0][_k].payload().inputDimPos_;

  if ((readMatchesTN.size() == 1) && (iPos == iPosTN) && (jPos == jPosTN))
    return std::make_tuple(true, iPosTN, jPosTN, kPosTN, GemmType::isTN);

  return std::make_tuple(false, 0, 0, 0, GemmType::isNN);
}

// is the access pattern C(i,j) - A(i,k) - B(k,j) in "reads"
// and C(i,j) in "writes"?
std::tuple<bool, int, int, int, GemmType> isGemmCorePattern(isl::ctx ctx,
  isl::union_map reads, isl::union_map writes) {
  using namespace matchers;
  auto _i = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto psWrite = allOf(access(_i, _j));
  auto writeMatches = match(writes, psWrite);

  if (writeMatches.size() != 1)
    return std::make_tuple(false, 0, 0, 0, GemmType::isNT);

  auto iPos = writeMatches[0][_i].payload().inputDimPos_;
  auto jPos = writeMatches[0][_j].payload().inputDimPos_; 
  auto matchedDims = lookUpPermutationOnReadGemm(ctx, reads, iPos, jPos);
  if (std::get<0>(matchedDims) == false)
    return std::make_tuple(false, 0, 0, 0, GemmType::isNT);

  iPos = std::get<1>(matchedDims);
  jPos = std::get<2>(matchedDims);
  auto kPos = std::get<3>(matchedDims);
  auto type = std::get<4>(matchedDims);
  return std::make_tuple(true, iPos, jPos, kPos, type);
}

// check access pattern for initialization statement in GEMM
// We expect C(i,j) = 0.0f
//
static bool hasGemmInitPatternImpl(
    const MappedScop& scop,
    isl::schedule_node leaf,
    MatMulInfo& mmi) {
  std::cout << __func__ << std::endl;
  auto schedule = leaf.get_prefix_schedule_union_map();
  if (!scheduleDimensionality(schedule, 2))
    return false;
  auto readsAndWrites = getReadsAndWritesForNode(scop, leaf);
  auto reads = readsAndWrites.first;
  auto writes = readsAndWrites.second;

  // we assume initialization statement with zero.
  // so we expect only a write.
  if ((reads.n_map() != 0) && (writes.n_map() != 1))
    return false;

  if (!_allOf(writes, isXd(2)))
    return false;

  auto writeMatches = isGemmInitPattern(leaf.get_ctx(), writes);
  if (!std::get<0>(writeMatches))
    return false;

  bool operation = false;
  try {
    operation = checkOperationsInInit(scop, leaf);
  } catch (...) {
    return false;
  }

  if (!operation)
    return false;

  mmi.writeToCInit = getArrayNameAndExtensionFromMap(
      writes,
      std::get<1>(writeMatches),
      std::get<2>(writeMatches));
  mmi.i = std::get<1>(writeMatches);
  mmi.j = std::get<2>(writeMatches);
  return true;
}

// check access pattern for core statement in GEMM
// We expect C(i,j) = C(i,j) + A(i,k) * B(k,j) _OR_
//  C(i,j) = C(i,j) + A(i,k) * B(j,k) _OR_
//  C(i,j) = C(i,j) + A(k,i) * B(k,j)
//
static bool hasGemmCorePatternImpl(
    const MappedScop& scop,
    isl::schedule_node leaf,
    MatMulInfo& mmi) {
  std::cout << __func__ << std::endl;
  auto schedule = leaf.get_prefix_schedule_union_map();
  if (!scheduleDimensionality(schedule, 3))
    return false;
  auto readsAndWrites = getReadsAndWritesForNode(scop, leaf);
  auto reads = readsAndWrites.first;
  auto writes = readsAndWrites.second;

  if ((reads.n_map() > 4) || (writes.n_map() != 1)) 
    return false;

  // TODO: check access dimensionality.

  auto matches = isGemmCorePattern(leaf.get_ctx(), reads, writes);
  if (!std::get<0>(matches))
    return false;
  auto iPos = std::get<1>(matches);
  auto jPos = std::get<2>(matches);
  auto kPos = std::get<3>(matches);
  auto type = std::get<4>(matches);

  auto _C_write = getArrayNameAndExtensionFromMap(
      writes, iPos, jPos);
  auto _C_read = getArrayNameAndExtensionFromMap(
      reads, iPos, jPos);
  
  if (_C_write != _C_read)
    return false;

  if ((mmi.i != iPos) ||
      (mmi.j != jPos))
    return false;

  if (mmi.writeToCInit != _C_write)
    return false;

  bool operation = false;
  try {
    operation = checkOperationsInGemmCore(scop, leaf);
  } catch (...) {
    return false;
  }

  if (!operation)
    return false;

  if (reads.n_map() == 4) {
    if (!_oneOf(reads, isConstant()))
      return false;
    else
      mmi.alpha = collectConstantName(reads);
  }

  mmi.isAtranspose = (type == GemmType::isTN) ? true : false;
  mmi.isBtranspose = (type == GemmType::isNT) ? true : false;  
  mmi.readFromC = _C_read;
  mmi.readFromA = getArrayNameAndExtensionFromMap(
      reads, iPos, kPos);
  mmi.readFromB = getArrayNameAndExtensionFromMap(
      reads, kPos, jPos);
  mmi.writeToC = _C_write;
  return true;
}

static isl::union_map dropBatchedDimOnAccesses(isl::union_map accesses) {

  assert(accesses.n_map() != 0 && "expect one or more accesses");

  auto v = vectorMapsFromUnionMap(accesses);
  for (auto& m: v) {  
    auto nameId = getName(m);
    if ((m.dim(isl::dim::in) < 1) || (m.dim(isl::dim::out) < 1))
      continue;
    m = m.project_out(isl::dim::in, 0, 1)
         .project_out(isl::dim::out, 0, 1);
    m = m.set_tuple_id(isl::dim::out, isl::id::alloc(m.get_ctx(), nameId, nullptr));
  }

  isl::union_map res = isl::union_map(v[0]);
  for (size_t i = 1; i < v.size(); i++) 
    res = res.unite(v[i]);  
  
  return res;
}

// check access pattern for initialization statement in batched GEMM
// We expect C(batchCount,i,j) = 0.0f
//
static bool hasBatchedGemmInitPatternImpl(const MappedScop& scop,
  isl::schedule_node leaf, BatchedMatMulInfo& bmmi) { 
  std::cout << __func__ << std::endl;
  auto schedule = leaf.get_prefix_schedule_union_map();
  if (!scheduleDimensionality(schedule, 3))
    return false;

  auto readsAndWrites = getReadsAndWritesForNode(scop, leaf);
  auto reads = readsAndWrites.first;
  auto writes = readsAndWrites.second;
  
  if ((reads.n_map() != 0) || (writes.n_map() != 1)) 
    return false;

  writes = dropBatchedDimOnAccesses(writes);

  auto matches = isGemmInitPattern(leaf.get_ctx(), writes);
  if (!std::get<0>(matches))
    return false;

  bool operation = false;
  try {
    operation = checkOperationsInInit(scop, leaf);
  } catch (...) {
    return false;
  }

  if (!operation)
    return false;
  
  bmmi.writeToC = getArrayNameAndExtensionFromMap(
       writes,
       std::get<1>(matches),
       std::get<2>(matches)); 
  bmmi.i = std::get<1>(matches);
  bmmi.i = std::get<2>(matches);
  return true;
}

static bool hasBatchedGemmCorePatternImpl(const MappedScop& scop,
  isl::schedule_node leaf, BatchedMatMulInfo& bmmi) {
  std::cout << __func__ << std::endl;
  auto schedule = leaf.get_prefix_schedule_union_map();
  if (!scheduleDimensionality(schedule, 4))
    return false;
  
  auto readsAndWrites = getReadsAndWritesForNode(scop, leaf);
  auto reads = readsAndWrites.first;
  auto writes = readsAndWrites.second;

  if ((reads.n_map() > 4) || (writes.n_map() != 1))
    return false;

  writes = dropBatchedDimOnAccesses(writes);
  reads = dropBatchedDimOnAccesses(reads);
 
  auto matches = isGemmCorePattern(leaf.get_ctx(), reads, writes);
  if (!std::get<0>(matches))
    return false;
  auto iPos = std::get<1>(matches);
  auto jPos = std::get<2>(matches);
  auto kPos = std::get<3>(matches);
  auto type = std::get<4>(matches);

  auto _C_write = getArrayNameAndExtensionFromMap(
    writes, iPos, jPos);
  auto _C_read = getArrayNameAndExtensionFromMap(
    reads, iPos, jPos);
  
  if (_C_write != _C_read)
    return false;

  bool operation = false;
  try {
    operation = checkOperationsInGemmCore(scop, leaf);
  } catch (...) { 
    return false;
  }

  if (!operation)
    return false;

  bmmi.isAtranspose = false;
  bmmi.isBtranspose = (type == GemmType::isNN) ? false : true;  
  bmmi.readFromC = _C_read;
  bmmi.readFromA = getArrayNameAndExtensionFromMap(
      reads, iPos, kPos);
  bmmi.readFromB = getArrayNameAndExtensionFromMap(
      reads, kPos, jPos);
  return true;
}

// debug function
//static isl::schedule_node printStatementsAndNodes(
//  const MappedScop& scop, isl::schedule_node node) {
//  if (node.isa<isl::schedule_node_leaf>()) {
//    std::cout << "NODE: " << node.to_str() << "\n";
//    std::cout << "STMT: " << getHalideStatement(scop.scop(), node);
//  }
//
//  for (int i = 0; i < node.n_children(); i++) {
//    node = printStatementsAndNodes(scop, node.child(i)).parent();
//  }
//  return node;
//}

// utility function.
//
// @param node: Root of the subtree to inspect.
// @param pattern: Pattern to look-up in the subtree.
// @param replacement: Replacement to be applied in case
// of a match with @p pattern.
static isl::schedule_node
replaceRepeatedly(isl::schedule_node node,
                   const matchers::ScheduleNodeMatcher &pattern,
                   const builders::ScheduleNodeBuilder &replacement) {

  while (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    node = rebuild(node, replacement);
  }
  return node;
}


// walk the schedule tree starting from "node" and in
// case of a match with the matcher "pattern" modify
// the schedule tree using the builder "replacement".
//
// @param node: Root of the subtree to inspect.
// @param pattern: Pattern to look-up in the subtree.
// @param replacement: Replacement to be applied in case of
// a match with @p pattern.
static isl::schedule_node replaceDFSPreorderRepeatedly(
    isl::schedule_node node, const matchers::ScheduleNodeMatcher &pattern,
    const builders::ScheduleNodeBuilder &replacement) {

  node = replaceRepeatedly(node, pattern, replacement);
  for (int i = 0; i < node.n_children(); i++) {
    node = replaceDFSPreorderRepeatedly(node.child(i), pattern, replacement)
               .parent();
  }
  return node;
}

// Squeeze the schedule tree.
// given a tree that looks like
//
// schedule (i)
//    schedule (j)
//      anyTree
//
// this will get simplify as
//
// schedule(i,j)
//   anyTree
//
// @param schedule_node: Current schedule node to be simplified.
isl::schedule_node squeezeTree(isl::schedule_node root) {

  isl::schedule_node parent, child, grandchild;
  auto matcher = [&]() {
    using namespace matchers;
    // clang-format off
    return band(parent,
      band(child,
        anyTree(grandchild)));
    //clang-format on
  }();

  auto merger = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    // clang-format off
    auto computeSched = [&]() {
      isl::multi_union_pw_aff sched =
        parent.band_get_partial_schedule().flat_range_product(
          child.band_get_partial_schedule());
      return sched;
    };
    // clang-format on
    auto st = [&]() { return subtreeBuilder(grandchild); };
    merger = band(computeSched, subtree(st));
  }

  root = replaceDFSPreorderRepeatedly(root, matcher, merger);
  return root.root();
}

std::vector<std::string> getStatementNames(std::vector<isl::map> maps) {
  std::vector<std::string> res{};
  for(const auto& m : maps) {
    isl::set dom = m.domain();
    assert(dom.has_tuple_name() && "expect tuple name");  
    res.push_back(dom.get_tuple_name());
  }
  return res;
}

__isl_give isl_schedule_node *distributeLoopsImpl(__isl_take isl_schedule_node* node,
    void* user) {
  isl::schedule_node nodeCpp = isl::manage(node);
  if (nodeCpp.isa<isl::schedule_node_mark>()) {
    std::string id = nodeCpp.mark_get_id().to_str();
    id = id.substr(0, id.find("@"));
    std::string substring = "_is_dominated";

    if (id.find(substring) != std::string::npos) {
      isl::id markNodeId = nodeCpp.mark_get_id();
      nodeCpp = nodeCpp.parent().parent();
      assert(nodeCpp.isa<isl::schedule_node_filter>());
      isl::union_set domPattern = nodeCpp.filter_get_filter();
      nodeCpp = nodeCpp.parent().parent();
      assert(nodeCpp.isa<isl::schedule_node_band>());
      nodeCpp = nodeCpp.order_before(domPattern);
      nodeCpp = nodeCpp.parent().previous_sibling().child(0);
      std::string::size_type i = id.find("_is_dominated");
      id.erase(i, substring.length());
      nodeCpp = nodeCpp.insert_mark(
        isl::id::alloc(nodeCpp.get_ctx(), id, markNodeId.get_user()));
      nodeCpp = nodeCpp.child(0).child(0);
      nodeCpp = nodeCpp.del();
    }
 
  } 
  return nodeCpp.release();
}

// distribute loop in the band which dominates the matcher's band.
// limitations:
// 1. This function work only with the matcher provided.
// 2. Moving the detected pattern before will not break any deps.
static isl::schedule_node distributeLoops(isl::schedule_node node) {
  node = isl::manage(isl_schedule_node_map_descendant_bottom_up(
    node.release(), distributeLoopsImpl, nullptr));
  return node.root();
}

// Is the matcher's band not dominated by an ancestor band
// which contains a superset of the matcher's band?
static bool isNotDominatedImpl(const MappedScop& scop,
    isl::schedule_node band) {
  if (!band.has_parent())
    return true;

  isl::schedule_node maybeFilter = band.parent();
  if (!maybeFilter.isa<isl::schedule_node_filter>()) 
    return true;

  isl::schedule_node filter = maybeFilter;
  isl::schedule_node sequence = filter.parent();
  if (!sequence.has_parent())
    return true;

  isl::schedule_node maybeBand = sequence.parent();
  if(!maybeBand.isa<isl::schedule_node_band>())
    return true;
  
  isl::schedule_node dominatorBand = maybeBand;
  isl::union_map dominatorBandPrefix = 
    dominatorBand.get_prefix_schedule_union_map();
  isl::union_map bandPrefix =
    band.get_prefix_schedule_union_map();

  if (bandPrefix.domain().is_subset(dominatorBandPrefix.domain())) {
    return false;
  }
  return true;
}

// is the current band too big for the CIM device?
// We have three options:
// 1. each dimension is smaller then TILE_FACTOR_CIM_DEVICE
// 2. 
//static bool needToTileImpl(const MappedScop& scop,
//    isl::schedule_node band) {
  //isl::union_set dom = scop.scop().domain(); 
  //isl::union_map prefixSchedule = band.get_prefix_schedule_union_map();
  //prefixSchedule = prefixSchedule.intersect_domain(dom);
  //auto vectorMaps = vectorMapsFromUnionMap(prefixSchedule);
//  return true;
//}

// tmp attempt to tiling
//static isl::schedule_node tileLoops(isl::schedule_node root) {
//  isl::schedule_node node = root.child(0).child(0);
//  isl::schedule_node_band band = node.as<isl::schedule_node_band>();  
//
//  isl::space space = band.get_space();
//  auto dims = space.dim(isl::dim::set);
//  auto sizes = isl::multi_val::zero(space);
//  
//  for (unsigned i = 0; i < dims; i++) 
//    sizes = sizes.set_val(i, isl::val(band.get_ctx(), 512));
//  
//  band = band.tile(sizes);
//
//  return band.root();
//}

static std::pair<isl::multi_union_pw_aff, isl::multi_union_pw_aff> tileNode(
    isl::schedule_node node) {
  isl::schedule_node_band band = node.as<isl::schedule_node_band>();
  
  isl::space space = band.get_space();
  auto dims = space.dim(isl::dim::set);
  auto sizes = isl::multi_val::zero(space);
  
  for (unsigned i = 0; i < dims; i++)
    sizes = sizes.set_val(i, isl::val(band.get_ctx(), TILE_FACTOR_CIM_DEVICE));
  band = band.tile(sizes);
  auto tileSchedule = band.get_partial_schedule();
  auto pointSchedule = band.child(0).band_get_partial_schedule();
  return std::make_pair(tileSchedule, pointSchedule);
} 
  

static isl::id getMarker(isl::ctx ctx, const std::string &label, const BlasInfo &bi) {
  if (label == "__tactics_mvt")
    return isl::id::alloc(ctx, label, new GemvInfo{bi.mvi});
  else if (label == "__tactics_gemm")
    return isl::id::alloc(ctx, label, new MatMulInfo{bi.mmi});
  else if (label == "__tactics_batched_gemm")
    return isl::id::alloc(ctx, label, new BatchedMatMulInfo{bi.bmmi});
  else if (label == "__tactics_gemm_is_dominated")
    return isl::id::alloc(ctx, label, new MatMulInfo{bi.mmi});
  else {
      std::cout << label << std::endl; 
      assert(0 && "case not defined");
  }
  return isl::id::alloc(ctx, "__tactics_undefined", nullptr);
}

  
// entry point for GEMV/GEMM/batched GEMM detection.

// The shape of the calculation for the supported matrix-vector
// multiplication includes an initialization of the output vector
// y with zeros and the actual core multiplications of the elements
//
// for (int i = 0; i <= M; i++) {
//  y[i] = 0.0f
//  for (int j = 0; j <= N; j++)
//    y[i] = y[i] + A[i][j] * x[i] _OR_
//    y[i] = y[i] + A[j][i] * x[i]
// }

// The shape of the calculation for the supported matrix
// multiplication includes an initialization of the output matrix
// C with zeros and the actual core multiplication of the elements
// of the input matrices A and B with a constant alpha:
//
//  for (int i = 0; i <= M; i++) {
//    for (int j = 0; j <= N; j++) {
//      C[i][j] = 0.000000f
//      for (int k = 0; k <= K; k++)
//        C[i][j] = C[i][j] +  A[i][k] * B[k][j] _OR_
//        C[i][j] = C[i][j] +  A[i][k] * B[j][k] _OR_
//        C[i][j] = C[i][j] + alpha * A[i][k] * B[k][j]
//     }
//   }

isl::schedule detectInSchedule(const MappedScop& scop) {
  isl::schedule schedule = toIslSchedule(scop.schedule());
  isl::schedule_node root = schedule.get_root();

  // only canonicalization pass.
  root = squeezeTree(root);

  BlasInfo bi;
  std::string labelNode = "__tactics_undefined";

  auto hasGemvInitPattern = [&](isl::schedule_node leaf) {
    leaf = leaf.parent().previous_sibling().child(0);
    return hasGemvInitPatternImpl(scop, leaf, bi.mvi);
  };

  auto hasGemvCorePattern = [&](isl::schedule_node leaf) {
    bool res = hasGemvCorePatternImpl(scop, leaf, bi.mvi);
    if (res)
      labelNode = "__tactics_mvt" + labelNode;
    return res;
  };

  auto hasGemmInitPattern = [&](isl::schedule_node leaf) {
    leaf = leaf.parent().previous_sibling().child(0);
    return hasGemmInitPatternImpl(scop, leaf, bi.mmi);
  };

  auto hasGemmCorePattern = [&](isl::schedule_node leaf) {
    bool res = hasGemmCorePatternImpl(scop, leaf, bi.mmi);
    if (res)
      labelNode = "__tactics_gemm" + labelNode;
    return res;
  };

  auto hasBatchedGemmInitPattern = [&](isl::schedule_node leaf) {
    leaf = leaf.parent().previous_sibling().child(0);
    return hasBatchedGemmInitPatternImpl(scop, leaf, bi.bmmi);
  };

  auto hasBatchedGemmCorePattern = [&](isl::schedule_node leaf) {
    bool res = hasBatchedGemmCorePatternImpl(scop, leaf, bi.bmmi);
    if (res)
      labelNode = "__tactics_batched_gemm" + labelNode;
    return res;
  };

  auto isNotDominated = [&](isl::schedule_node band) {
    bool res = isNotDominatedImpl(scop, band);
    if (!res) 
      labelNode = "_is_dominated";
    else 
      labelNode = ""; 
    return true;
  };

  isl::schedule_node scheduleBody, filterInit, filterCore;
  auto matcher = [&]() {
    using namespace matchers;
    // clang-format off
    return
    band(isNotDominated, scheduleBody, 
      sequence(
        filter(filterInit, leaf()),
        filter(filterCore, leaf(_or(
                        _and(hasBatchedGemmInitPattern, hasBatchedGemmCorePattern),
                        _and(hasGemmInitPattern, hasGemmCorePattern),
                        _and(hasGemvInitPattern, hasGemvCorePattern))))));
    // clang-format on
  }();

  auto builder = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto tileScheduler = [&]() {
      auto descr = BandDescriptor(scheduleBody);
      auto tileSchedule = tileNode(scheduleBody).first;
      descr.partialSchedule = tileSchedule;
      return descr;
    };
    auto pointScheduler = [&]() {
      auto descr = BandDescriptor(scheduleBody);
      auto pointSchedule = tileNode(scheduleBody).second;
      descr.partialSchedule = pointSchedule;
      return descr;
    };
    auto marker = [&]() {
      return getMarker(scheduleBody.get_ctx(), labelNode, bi);
    };
    auto getfilterInitialization = [&]() {
      return filterInit.filter_get_filter();
    };
    auto getfilterCore = [&]() {
      return filterCore.filter_get_filter();
    };
    builder = band(tileScheduler,
                mark(marker,
                  band(pointScheduler,
                    sequence(
                      filter(getfilterInitialization),
                      filter(getfilterCore)))));
  }
                
  if(!FLAGS_disable_tactics) {
    //root = wrapOnMatch(root, matcher, labelNode, bi).root();
    root = rebuildOnMatch(root, matcher, builder).root();
    root = distributeLoops(root).root();
    //root = tileLoops(root);
    std::cout << root.to_str() << "\n";
  }

  return root.get_schedule();
}

} // namespace tactics
} // namespace polyhedral
} // namespace tc
