#include "tc/core/polyhedral/tactics/tactics_optimizer.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/matchers/matchers.h"
#include "tc/core/polyhedral/matchers/access_patterns.h"
#include "tc/core/polyhedral/matchers/access.h"

namespace tc {
namespace polyhedral {
namespace tactics {

using namespace matchers;

// Finds all nodes in a schedule tree rooted at root that match m
std::vector<isl::schedule_node>
findPatterns(const matchers::ScheduleNodeMatcher &m,
             isl::schedule_node root) {

  std::vector<isl::schedule_node> rootMatched;
  std::stack<isl::schedule_node> nodeStack;
  nodeStack.push(root);

  while(nodeStack.empty() == false) {
    root = nodeStack.top();
    nodeStack.pop();

    if(matchers::ScheduleNodeMatcher::isMatching(m, root)) {
      rootMatched.push_back(root);
    }
  
    size_t n_children =
      static_cast<size_t>(isl_schedule_node_n_children(root.get()));
    for(size_t i = 0; i < n_children; i++) {
      nodeStack.push(root.child(i));
    }
  }

  return rootMatched;
}

// Checks if the node n has already been marked by a matcher
bool hasTacticsMarker(isl::schedule_node n)
{
  if(!n.has_parent() || !n.parent().isa<isl::schedule_node_mark>())
    return false;

  isl::schedule_node_mark mark = n.parent().as<isl::schedule_node_mark>();

  return mark.get_id().get_name().rfind("__tactics", 0) == 0;
}

// Checks if the node n or any of its ancestors up to the root have
// already been marked by a matcher
bool selfOrAncestorHasTacticsMarker(isl::schedule_node n) {
  if(hasTacticsMarker(n))
    return true;

  if(n.has_parent())
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
  explicit ScopedCtx(isl::ctx &&ctx) : ctx(ctx) {}
  ScopedCtx(const ScopedCtx &) = delete;
  ScopedCtx(ScopedCtx &&) = default;
  ~ScopedCtx() { isl_ctx_free(ctx.release()); }

  ScopedCtx &operator=(const ScopedCtx &) = delete;
  ScopedCtx &operator=(ScopedCtx &&) = default;

  operator isl::ctx() { return ctx; }
  operator isl_ctx *() { return ctx.get(); }

private:
  isl::ctx ctx;
};

bool operator==(const Halide::Internal::Variable& a,
		const Halide::Internal::Variable& b)
{
  return a.name == b.name;
}

bool operator!=(const Halide::Internal::Variable& a,
		const Halide::Internal::Variable& b)
{
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
      std::equal(dims.begin(), dims.end(),
		 b.dims.begin(),
		 [](const Halide::Internal::Variable* const va,
		    const Halide::Internal::Variable* const vb) -> bool
		 { return *va == *vb; });
  }

  bool operator!=(const TensorAccess& b) const {
    return !(*this == b);
  }
};

std::ostream& operator<<(std::ostream& os, const Halide::Internal::Variable& v) {
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
bool extractDimBounds(isl::set set,
		      const std::string& dimName,
		      int& min,
		      int& max)
{
  isl::space space = set.get_space();

  int dimIdx = space.find_dim_by_name(isl::dim::out, dimName);

  if(dimIdx == -1)
    return false;
  
  // Extract constants for bounds from piecewise quasi-affine
  // expressions for the minimum and maximum
  isl::pw_aff minAff = set.dim_min(dimIdx);
  isl::pw_aff maxAff = set.dim_max(dimIdx);

  // Check that the expressions are all constant and constructed of a
  // single piece
  if (!minAff.is_cst() || !maxAff.is_cst() ||
      minAff.n_piece() != 1 || maxAff.n_piece() != 1)
    {
      return false;
    }

  minAff.foreach_piece(
	[&](isl::set s, isl::aff a) {
	  min = std::stoi(a.get_constant_val().to_str());
	});

  maxAff.foreach_piece(
	[&](isl::set s, isl::aff a) {
	  max = std::stoi(a.get_constant_val().to_str())+1;
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
    // The shape of the calculation for the supported matrix
    // multiplication includes an initialization of the output matrix
    // C with zeros and the actual core multiplication of the elements
    // of the input matrices A and B with a constant alpha:
    //
    //   for (int i = 0; i <= M; i++) {
    //     for (int j = 0; j <= N; j++) {
    //       C[i][j] = 0.000000f;
    //
    //       for (int k = 0; k <= K; k++)
    //         C[i][j] = C[i][j] + alpha * A[i][k] * B[k][j];
    //     }
    //   }

    return band(
		band(
		     sequence(filter(leaf()),
			      filter(leaf()))));
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

      if(!v)
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
  static bool extractTensorAccess0D(const Halide::Expr& expr,
   				    TensorAccess& ret)
  {
    return extractTensorAccessND<0>(expr, ret);
  }

  // Extracts the components of a 2D tensor access from an expression
  // (e.g., C(i, j)) if expr is a 2D tensor access. Otherwise the
  // function returns false.
  static bool extractTensorAccess2D(const Halide::Expr& expr,
   				    TensorAccess& ret)
  {
    return extractTensorAccessND<2>(expr, ret);
  }

  // Extracts all operands of a multiplication of the form (a * (b *
  // (c * (...))) in order from expr.
  static void extractMultiplicationOperands(
	const Halide::Expr& expr,
	std::vector<const Halide::Expr*>& operands)
  {
    const Halide::Internal::Mul* mul;

    if((mul = expr.as<Halide::Internal::Mul>())) {
      extractMultiplicationOperands(mul->a, operands);
      extractMultiplicationOperands(mul->b, operands);
    } else {
      operands.push_back(&expr);
    }
  }

  // Returns the Halide statement associated to a schedule node
  // n. Throws if no Halide statement is associated with the node.
  static const Halide::Internal::Stmt&
  getHalideStatement(const Scop& scop, isl::schedule_node n)
  {
    auto dom = n.get_domain();
    isl::set_list instances = dom.get_set_list();
    isl::set instance = instances.get_at(0);
    isl::id instId = instance.get_tuple_id();

    return scop.halide.statements.at(instId);
  }
  
  // Extracts the operands of the matrix multiplication from the root
  // of the match obtained from the matcher. On success, the operands
  // are provided in mmi and the function return true. If the
  // operations and accesses in the subtree do not match a matrix
  // multiplication, the function returns false.
  static bool extractMatMulInfo(isl::schedule_node m,
				const MappedScop& scop,
				MatMulInfo& mmi)
  {
    auto& lscop = scop.scop();

    // The ID of the leaf node of the init statement maps to the halide statement
    isl::schedule_node initLeaf = m.child(0).child(0).child(0).child(0);
    const Halide::Internal::Stmt& initStatement =
      getHalideStatement(lscop, initLeaf);

    // The init statement should be of the form
    //
    //   C(m, k) = 0.000000f
    const Halide::Internal::Provide* initProvide =
      initStatement.as<Halide::Internal::Provide>();

    if(!initProvide || initProvide->args.size() != 2)
      return false;

    TensorAccess accC_init;
    accC_init.tensor = initProvide->name;
    accC_init.dims = {
	initProvide->args[0].as<Halide::Internal::Variable>(),
	initProvide->args[1].as<Halide::Internal::Variable>()
    };

    int minM_init, maxM_init;
    int minK_init, maxK_init;

    isl::set initSet = initLeaf.get_domain().get_set_list().get_at(0);

    if(!extractDimBounds(initSet, accC_init.dims[0]->name, minM_init, maxM_init) ||
       !extractDimBounds(initSet, accC_init.dims[1]->name, minK_init, maxK_init))
      {
	return false;
      }

    if(minM_init != 0 || minK_init != 0)
      return false;

    // RHS of initilization must be a single constant
    if(initProvide->values.size() != 1)
      return false;

    const Halide::Internal::FloatImm* initVal =
      initProvide->values[0]. as<Halide::Internal::FloatImm>();

    if(!initVal)
      return false;

    if(initVal->value != 0.0)
      return false;

    // The ID of the leaf node of the core statement maps to the halide statement
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

    if(!coreProvide || coreProvide->args.size() != 2)
      return false;

    // Extract 2D tensor access from provide node
    TensorAccess accC_LHS;
    accC_LHS.tensor = coreProvide->name;
    accC_LHS.dims = {
	coreProvide->args[0].as<Halide::Internal::Variable>(),
	coreProvide->args[1].as<Halide::Internal::Variable>()
    };

    if(!accC_LHS.dims[0] || !accC_LHS.dims[1])
      return false;

    // RHS must be a single call to ReductionUpdate
    if(coreProvide->values.size() != 1)
      return false;
   
    const Halide::Internal::Call* reductionCall =
      coreProvide->values[0].as<Halide::Internal::Call>();

    if(!reductionCall ||
       !reductionCall->is_intrinsic("ReductionUpdate") ||
       reductionCall->args.size() != 1)
      {
	return false;
      }

    // Check that the reduction is an addition
    const Halide::Internal::Add* addition =
      reductionCall->args[0].as<Halide::Internal::Add>();

    if(!addition)
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

    if(extractTensorAccess2D(addition->a, accC_RHS)) {
      multiplication = &addition->b;
    } else if(extractTensorAccess2D(addition->b, accC_RHS)) {
      multiplication = &addition->a;
    } else {
      return false;
    }

    // Check that matrix access on the LHS and RHS reference same
    // matrix and same dimensions
    if(accC_LHS != accC_RHS)
      return false;

    // Extract input matrices A and B from multiplication
    std::vector<const Halide::Expr*> mulOperands;

    extractMultiplicationOperands(*multiplication, mulOperands);

    if(mulOperands.size() != 3)
      return false;

    // Operands must be two 2D tensor accesses and a constant
    std::vector<TensorAccess> accAB;
    TensorAccess accAlpha;
    bool hasAlpha = false;
    
    for(const Halide::Expr* mulOperand: mulOperands) {
      TensorAccess tmp;

      if(extractTensorAccess2D(*mulOperand, tmp))
	accAB.push_back(tmp);
      else if(extractTensorAccess0D(*mulOperand, tmp)) {
	hasAlpha = true;
	accAlpha = tmp;
      }
    }

    if(accAB.size() != 2 || !hasAlpha)
      return false;

    // Output Matrix C must not appear again in the multiplication
    if(accAB[0].tensor == accC_LHS.tensor ||
       accAB[1].tensor == accC_LHS.tensor)
      {
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
    if(*accAB[0].dims[0] == *accAB[1].dims[1]) {
      accA = &accAB[1];
      accB = &accAB[0];
    } else if(*accAB[1].dims[0] == *accAB[0].dims[1]) {
      accA = &accAB[0];
      accB = &accAB[1];
    } else {
      return false;
    }

    // Then, check if the dimensions correspond to those of the output
    // matrix C
    if(*accA->dims[0] != *accC_LHS.dims[0] ||
       *accB->dims[1] != *accC_LHS.dims[1])
      {
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

    if(!extractDimBounds(coreSet, accA->dims[0]->name, minM, maxM) ||
       !extractDimBounds(coreSet, accA->dims[1]->name, minN, maxN) ||
       !extractDimBounds(coreSet, accB->dims[1]->name, minK, maxK))
      {
	return false;
      }

    if(minM != 0 || minN != 0 || minK != 0)
      return false;
    
    // The access to the output matrix of the core statement must
    // match the one of the initilization statement
    if(accC_LHS.tensor != accC_init.tensor ||
       maxM != maxM_init ||
       maxK != maxK_init)
      {
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
  static bool checkMatch(isl::schedule_node m, const MappedScop& scop)
  {
    MatMulInfo mmi;

    return extractMatMulInfo(m, scop, mmi);
  }

  // Processes a match: Marks the root node of the match as a matrix
  // multiplication and adds the matrix multiplication information to
  // the replacements to be performed upon code generation.
  static isl::schedule_node processMatch(isl::schedule_node m,
					 const MappedScop& scop,
					 TacticsReplacements& replacements)
  {
    isl::id markId = isl::id::alloc(m.get_ctx(), "__tactics_gemm", nullptr);
    MatMulInfo mmi;

    if(extractMatMulInfo(m, scop, mmi))
      replacements.matmul.emplace(std::make_pair(markId, mmi));

    return m.insert_mark(markId);
  }
};

isl::schedule optimizeGemmSchedule(const MappedScop& scop,
				   TacticsReplacements& replacements)
{
  isl::schedule schedule = toIslSchedule(scop.schedule());
  isl::schedule_node root = schedule.get_root();
  
  auto matcherGEMM = GemmOptimizer::getMatcher();


  bool restart;

  do {
    restart = false;
    std::vector<isl::schedule_node> matches = findPatterns(matcherGEMM, root);
    
    for(auto& m: matches) {
      // Make sure that the same subtree is not matched twice and that
      // matches do not overlap
      if(selfOrAncestorHasTacticsMarker(m))
	continue;
      
      if(GemmOptimizer::checkMatch(m, scop)) {
	root = GemmOptimizer::processMatch(m, scop, replacements);

	// When processing a match previous nodes might get
	// invalidated -> restart matching
	restart = true;
	break;
      }
    }
  } while(restart);

  schedule = root.get_schedule();

  return schedule;
}

} // namespace tactics
} // namespace polyhedral
} // namespace tc
