/**
 * Copyright (c) 2017-2018, Facebook, Inc.
 * Copyright (c) 2019-present, Inria
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "tc/core/check.h"
#include "tc/core/cuda/cuda_libraries.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/body.h"
#include "tc/core/polyhedral/codegen.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/polyhedral/tactics/codegen.h"
#include "tc/core/polyhedral/tactics/tactics_optimizer.h"

using namespace std;

using tc::polyhedral::detail::ScheduleTreeContext;
using tc::polyhedral::detail::ScheduleTreeDomain;
using tc::polyhedral::detail::toIslSchedule;

namespace tc {
namespace polyhedral {
namespace tactics {

namespace {

static std::string halideTypeString(const Halide::Type& t) {
  if (t.is_bool()) {
    return "bool";
  } else if (t.is_int() && t.bits() == 8) {
    return "char";
  } else if (t.is_int() && t.bits() == 16) {
    return "short";
  } else if (t.is_int() && t.bits() == 32) {
    return "int";
  } else if (t.is_int() && t.bits() == 64) {
    return "long";
  } else if (t.is_uint() && t.bits() == 8) {
    return "unsigned char";
  } else if (t.is_uint() && t.bits() == 16) {
    return "unsigned short";
  } else if (t.is_uint() && t.bits() == 32) {
    return "unsigned int";
  } else if (t.is_uint() && t.bits() == 64) {
    return "unsigned long";
  } else if (t.is_float() && t.bits() == 16) {
    return "half";
  } else if (t.is_float() && t.bits() == 32) {
    return "char";
  } else if (t.is_float() && t.bits() == 64) {
    return "double";
  }
  std::stringstream ss;
  ss << t;
  return ss.str();
}

struct WS {
  static thread_local int n;
  WS() {
    n += 2;
  }
  ~WS() {
    n -= 2;
  }
  string tab() {
    stringstream ss;
    for (int i = 0; i < n; ++i) {
      ss << " ";
    }
    return ss.str();
  }
};
thread_local int WS::n = 0;

std::string makePointerName(std::string n) {
  return string("p") + n;
}

std::string makeName(std::string n) {
  return n;
}

std::string makeReductionTmpName(isl::id updateId, const Scop& scop) {
  int pos = scop.reductionUpdatePos(updateId);
  return "acc_" + std::to_string(pos);
}

template <typename T>
inline vector<T> operator+(vector<T> a, const vector<T>& b) {
  vector<T> res{a};
  res.insert(res.begin() + res.size(), b.begin(), b.end());
  return res;
}

struct AstPrinter {
 public:
  AstPrinter(const CodegenContext& context) : context_(context) {}
  void emit(isl::ast_node node) {
    emitAst(node);
  }

 private:
  void emitFor(isl::ast_node_for node);
  void emitIf(isl::ast_node_if node);
  void emitStmt(isl::ast_node_user node);
  void emitAst(isl::ast_node node);
  void emitMark(isl::ast_node_mark mark);
  void emitMatmulMark(isl::ast_node_mark mark);
  void emitGemvMark(isl::ast_node_mark mark);
  void emitBatchedMatmulMark(isl::ast_node_mark mark);

 private:
  const CodegenContext& context_;
  // Identifier of reduction update node processed by emitStmt for use
  // in a tree synchronization in a subsequent call to emitStmt.
  isl::id reductionUpdateNodeId_;
  // Has a reduction init statement been encountered in a previous
  // call to emitStmt without a subsequent tree synchronization?
  bool inReduction_ = false;
};

vector<pair<string, string>> emitParams(const Scop& scop) {
  vector<pair<string, string>> res;
  res.reserve(scop.halide.params.size());
  // Halide params. One of these two vectors will be empty.
  for (auto p : scop.halide.params) {
    string sname = p.name();
    string stype = halideTypeString(p.type());

    res.push_back(make_pair(sname, stype));
  }

  return res;
}

// Returns a pair (tensor name, tensor type)
pair<string, string> emitTypedTensorName(
    Halide::OutputImageParam t,
    bool constInput = false,
    std::string (name_fun)(std::string) = makePointerName) {
  stringstream sstype;
  sstype << (constInput ? "const " : "") << halideTypeString(t.type()) << "*";

  string sname = name_fun(t.name());

  return make_pair(sname, sstype.str());
}

vector<pair<string, string>> emitTypedTensorNames(
    const vector<Halide::OutputImageParam>& tensors,
    std::string (name_fun)(std::string) = makePointerName) {
  vector<pair<string, string>> res;
  res.reserve(tensors.size());
  for (auto t : tensors) {
    res.push_back(emitTypedTensorName(t, false, name_fun));
  }
  return res;
}

vector<pair<string, string>> emitTypedTensorNames(
    const vector<Halide::ImageParam>& tensors,
    std::string (name_fun)(std::string) = makePointerName) {
  vector<pair<string, string>> res;
  res.reserve(tensors.size());
  for (auto t : tensors) {
    res.push_back(emitTypedTensorName(t, true, name_fun));
  }
  return res;
}

void emitArgs(stringstream& ss, const Scop& scop) {
  // Order is: params, outs, ins
  auto sigVec = emitParams(scop);
  sigVec = sigVec + emitTypedTensorNames(scop.halide.outputs);
  sigVec = sigVec + emitTypedTensorNames(scop.halide.inputs);
  for (auto& s : sigVec) {
    ss << s.second << " " << s.first;
    if (s != sigVec.back()) {
      ss << ", ";
    }
  }
}

void emitKernelSignature(
    stringstream& ss,
    const std::string& specializedName,
    const Scop& scop) {
  TC_CHECK_NE(specializedName, "") << "name not provided";
  ss << "void " << specializedName << "(";
  emitArgs(ss, scop);
  ss << ") {" << endl;
}

// This is similar to the pass unpack_buffers in
// Halide, which unpacks strides, grabs alignment constraints,
// etc.
// TODO: this is still incorrect because at this point we only use the
// DLTensor shape (i.e. sizes) of the computations.
// To be correct we need the strides.
// Unfortunately, strides are related to memory allocation and are ML
// framework specific.
// Halide has its own facilities to allocate memory and handles concrete
// allocated memory at the (linearized) Buffer level.
// We don't want that, and we are even at a higher level of IR where Buffer to
// not exist.
// So we must pass an additional structure to save strides that we collect at
// runtime from the actual tensors that are passed to TcOp.
// We could go parametric but then we need to pass all the strides as
// parameters to the kernel call. This is doable, we've been doing it since
// day 1 with fbcuda's DeviceTensor but it loses runtime alignment information
// (or we need to jump through hoops to make proper use of it).
// So the better path here is probably to JIT everything, except people want
// as parametric code as possible, **sigh**.
void emitTensorView(
    stringstream& ss,
    Halide::OutputImageParam p,
    const map<string, Halide::Expr>& paramValues,
    bool constInput = false) {
  WS ws;
  stringstream ssViewType;
  for (int i = 1; i < p.dimensions(); ++i) { // Skip the outermost dimension
    Halide::Expr extent = p.parameter().extent_constraint(i);
    extent = Halide::Internal::substitute(paramValues, extent);
    TC_CHECK(extent.defined())
        << "Undefined extent on input/output tensor. Forward bounds inference should have set these\n";
    ssViewType << "[" << extent << "]";
  }
  ss << ws.tab();
  ss << (constInput ? "const " : "") << halideTypeString(p.type()) << " (*"
     << p.name() << ")" << ssViewType.str();
  ss << " = ";
  ss << "(" << (constInput ? "const " : "") << halideTypeString(p.type())
     << " (*)" << ssViewType.str() << ")" << makePointerName(p.name()) << ";";
  ss << endl;
}

void emitTensorViews(
    stringstream& ss,
    const vector<Halide::OutputImageParam>& params,
    const map<string, Halide::Expr>& paramValues) {
  for (auto p : params) {
    emitTensorView(ss, p, paramValues);
  }
}

void emitTensorViews(
    stringstream& ss,
    const vector<Halide::ImageParam>& params,
    const map<string, Halide::Expr>& paramValues) {
  for (auto p : params) {
    emitTensorView(ss, p, paramValues, true);
  }
}

void AstPrinter::emitFor(isl::ast_node_for node) {
  WS ws;
  context_.ss << ws.tab();
  string iter = node.get_iterator().to_C_str();
  context_.ss << "for (int " << iter << " = " << node.get_init().to_C_str()
              << "; " << node.get_cond().to_C_str() << "; " << iter
              << " += " << node.get_inc().to_C_str() << ") {" << endl;
  emitAst(node.get_body());
  context_.ss << ws.tab() << "}" << endl;
}

void AstPrinter::emitIf(isl::ast_node_if node) {
  WS ws;
  context_.ss << ws.tab();
  context_.ss << "if (" << node.get_cond().to_C_str() << ") {" << endl;
  emitAst(node.get_then());
  context_.ss << ws.tab() << "}";
  if (node.has_else()) {
    context_.ss << " else {" << endl;
    emitAst(node.get_else());
    context_.ss << ws.tab() << "}";
  }
  context_.ss << endl;
}

void emitUserStmt(isl::id stmtId, const CodegenStatementContext& context) {
  TC_CHECK(context.scop().halide.statements.count(stmtId))
      << "No stmt with id " << stmtId << "\n";
  auto provide = context.scop().halide.statements.at(stmtId);

  auto op = provide.as<Halide::Internal::Provide>();

  if (op) {
    TC_CHECK(op) << "Expected a Provide node: " << provide << '\n';
    detail::emitMappedTensorAccess(op->name, op, op->args, context);
    context.ss << " = ";
    TC_CHECK(op->values.size() == 1)
        << "Multi-valued Provide: " << Halide::Internal::Stmt(provide) << "\n";
    detail::emitHalideExpr(op->values[0], context);
    context.ss << ";" << endl;
  }

  auto opCall = provide.as<Halide::Internal::Call>();

  if (opCall) {
    detail::emitHalideExpr(opCall, context);
  }
}

void emitReductionUpdate(
    isl::id stmtId,
    const CodegenStatementContext& context) {
  // This is a Halide reduction. The reduction update is stored as a
  // recursive expression (e.g. f(x, y) = f(x, y) + foo). Replace
  // the recursive call with a variable representing the temporary
  // accumulator. It's probably at the root of the expression tree,
  // but it's easy enough to be generic here to accommodate more
  // complex reductions in the future.
  string tmp = makeReductionTmpName(stmtId, context.scop());
  context.ss << tmp << " = ";
  auto provide = context.scop()
                     .halide.statements.at(stmtId)
                     .as<Halide::Internal::Provide>();
  Halide::Expr rhs = provide->values[0];
  map<string, string> substitutions;
  substitutions[provide->name] = tmp;
  detail::emitHalideExpr(rhs, context, substitutions);
  context.ss << ";" << endl;
}

void emitReductionInit(
    isl::id stmtId,
    isl::id updateId,
    const CodegenContext& context) {
  // Emit the identity of a reduction, to initialize a local accumulator.
  auto provide = context.scop()
                     .halide.statements.at(updateId)
                     .as<Halide::Internal::Provide>();
  context.ss << makeReductionTmpName(updateId, context.scop()) << " = ";
  auto call = provide->values[0].as<Halide::Internal::Call>();
  TC_CHECK(call && call->is_intrinsic(tc2halide::kReductionUpdate));
  auto assoc = prove_associativity(provide->name, provide->args, call->args);
  if (!assoc.associative()) {
    std::stringstream ss;
    ss << "Not associative: " << provide->name << ", provide: ";
    Halide::Internal::IRPrinter p(ss);
    p.print(Halide::Internal::Stmt(provide));
    ss << "\nGenerated so far:\n" << context.ss.str();
    throw codegen::NotAssociativeError(ss.str());
  }
  auto statementContext = CodegenStatementContext(context, stmtId);
  TC_CHECK_EQ(assoc.pattern.identities.size(), 1u);
  detail::emitHalideExpr(assoc.pattern.identities[0], statementContext);
  context.ss << ";" << endl;
}

namespace {
template <typename AFF>
isl::ast_expr buildAccess(AFF access, const CodegenStatementContext& context) {
  return context.build().access_from(access);
}

void emitAccess(isl::ast_expr access, const CodegenStatementContext& context) {
  context.ss << access.to_C_str();
}

template <typename AFF>
void emitAccess(AFF access, const CodegenStatementContext& context) {
  emitAccess(buildAccess(access, context), context);
}

// Check that the given expression is an access with constant index expressions
void checkConstantAccess(isl::ast_expr expr) {
  auto op = expr.as<isl::ast_expr_op>();
  auto access = op.as<isl::ast_op_access>();
  TC_CHECK(access);
  for (int i = 1; i < access.get_n_arg(); ++i) {
    auto arg = access.get_arg(i);
    TC_CHECK(arg.as<isl::ast_expr_int>())
        << "expected constant subscript, got " << arg.to_C_str();
  }
}

// Print an access to a(n array of) register(s), checking that
// the index expressions are constant.
void emitRegisterAccess(
    isl::pw_multi_aff access,
    const CodegenStatementContext& context) {
  auto expr = buildAccess(access, context);
  checkConstantAccess(expr);
  emitAccess(expr, context);
}

// Print an access to global memory, wrapping the access in an "__ldg()"
// call if the accessed tensor is known to be read-only.
void emitGlobalAccess(
    isl::multi_pw_aff access,
    const CodegenStatementContext& context) {
  emitAccess(access, context);
}
} // namespace

void emitCopyStmt(const CodegenStatementContext& context) {
  auto stmtId = context.statementId();

  auto iteratorMap = context.iteratorMap();
  auto promoted = iteratorMap.range_factor_range();
  auto original = iteratorMap.range_factor_domain().range_factor_range();
  auto isRead = stmtId.get_name() == kReadIdName;

  if (isRead) {
    emitAccess(isl::multi_pw_aff(promoted), context);
    context.ss << " = ";
    emitGlobalAccess(isl::multi_pw_aff(original), context);
  } else {
    emitGlobalAccess(isl::multi_pw_aff(original), context);
    context.ss << " = ";
    emitAccess(isl::multi_pw_aff(promoted), context);
  }
  context.ss << ";" << std::endl;
}

void AstPrinter::emitStmt(isl::ast_node_user node) {
  isl::ast_expr_op usrExp = node.get_expr().as<isl::ast_expr_op>();
  auto stmtId = usrExp.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto nodeId = node.get_annotation();
  auto statementContext = CodegenStatementContext(context_, nodeId);
  TC_CHECK_EQ(context_.nodeInfoMap.count(nodeId), 1u)
      << "no info for node " << nodeId;

  WS ws;
  context_.ss << ws.tab();

  if (context_.scop().isTreeSyncId(stmtId)) {
    TC_CHECK_EQ(0, 1) << "NYI";
  } else if (context_.scop().isDefaultReductionInitId(stmtId)) {
    auto updateId = context_.scop().getReductionUpdateForDefaultInit(stmtId);
    emitReductionInit(stmtId, updateId, context_);
    inReduction_ = true;
  } else if (inReduction_ && context_.scop().isReductionUpdate(stmtId)) {
    emitReductionUpdate(stmtId, statementContext);
    reductionUpdateNodeId_ = nodeId;
  } else if (context_.scop().isSyncId(stmtId)) {
    TC_CHECK_EQ(0, 1) << "NYI";
  } else if (context_.scop().isWarpSyncId(stmtId)) {
    TC_CHECK_EQ(0, 1) << "NYI";
  } else if (
      stmtId.get_name() == kReadIdName || stmtId.get_name() == kWriteIdName) {
    emitCopyStmt(statementContext);
  } else { // regular statement
    auto mappedStmtId = statementContext.statementId();
    TC_CHECK_EQ(stmtId, mappedStmtId)
        << "statement ids in expr (" << stmtId << ") and in iteratorMaps ("
        << mappedStmtId << ") do not match";
    emitUserStmt(stmtId, statementContext);
  }
}

// create GEMM blas call.
// cimblas_gemm(
//  transA : is A transpose?
//  transB : is B transpose?
//  m : specifies the number of rows for A and C
//  n : specifies the number of cols for B and C
//  k : specifies the number of cols for A and the number of rows for B
//  alpha : scalar alpha
//  A : 2-d tensor
//  lda : max(1, k) if no trans or max(1, m)
//  B : 2-d tensor
//  ldb : max(1, n) if no trans or max(1, k)
//  beta : scalar beta
//  C : 2-d tensor
//  ldc : max(1,m));
void AstPrinter::emitMatmulMark(isl::ast_node_mark mark) {
  isl::id markId = mark.get_id();
  void* user = isl_id_get_user(markId.get());
  MatMulInfo* payload = static_cast<MatMulInfo*>(user);

  WS ws;

  int m = 
    payload->readFromA.dims[0].ub - payload->readFromA.dims[0].lb;
  int n =
    payload->readFromB.dims[1].ub - payload->readFromB.dims[1].lb;
  int k =
    payload->readFromA.dims[1].ub - payload->readFromA.dims[1].lb;
  int lda = (payload->isAtranspose) ? std::max(1, m) : std::max(1, k);
  int ldb = (payload->isBtranspose) ? std::max(1, k) : std::max(1, n);
  int ldc = std::max(1, m);

  context_.ss << ws.tab() << "cimblas_gemm("
              << payload->isAtranspose << ", "
              << payload->isBtranspose << ", " 
              << m << ", " << n << ", " << k << ", "
              << payload->alpha << ", "
              << payload->readFromA.name << ", "
              << lda << ", "
              << payload->readFromB.name << ", " 
              << ldb << ", "
              << payload->beta << ", " 
              << payload->writeToC.name << ", "  
              << ldc << ");" << std::endl;

  delete payload;
}

// create GEMV blas call.
// cimblas_gemv(
//   transA : is A transpose?
//   m : specifies the number of rows for A
//   n : specifies the number of cols for A
//   alpha : sclara alpha
//   A : 2-d tensor
//   lda : max(1, n)
//   X : 1-d tensor
//   incx : increment for X
//   beta : scalar beta
//   Y : 1-d tensor
//   incy : increment for y)
void AstPrinter::emitGemvMark(isl::ast_node_mark mark) {
  isl::id markId = mark.get_id();
  void* user = isl_id_get_user(markId.get());
  GemvInfo* payload = static_cast<GemvInfo*>(user);

  WS ws;

  int m = payload->readFromA.dims[0].ub - payload->readFromA.dims[0].lb;
  int n = payload->readFromA.dims[1].ub - payload->readFromA.dims[0].lb;
  int lda = std::max(1, n);

  context_.ss << ws.tab() << "cim_gemv(" 
              << payload->isAtranspose << ", "
              << m << ", " << n << ", "
              << payload->alpha << ", "
              << payload->readFromA.name << ", "
              << lda << ", "
              << payload->readFromX.name << ", "
              << payload->incx << ", "
              << payload->beta << ", "
              << payload->writeToY.name << ", "
              << payload->incy << ");" << std::endl;

  delete payload;
}

void AstPrinter::emitBatchedMatmulMark(isl::ast_node_mark mark) {
  isl::id markId = mark.get_id();
  void* user = isl_id_get_user(markId.get());
  BatchedMatMulInfo* payload = static_cast<BatchedMatMulInfo*>(user);

  WS ws;

  context_.ss << ws.tab() << "cim_batched_gemm(" << payload->writeToC.name << ","
              << payload->readFromA.name << "," << payload->readFromB.name << "," 
              << payload->alpha << ");"
              << std::endl;

  delete payload; 
}

void AstPrinter::emitMark(isl::ast_node_mark mark) {
  isl::id markId = mark.get_id();
  const std::string markType = markId.get_name();

  if (markType == "__tactics_gemm") {
    emitMatmulMark(mark);
  } 
  else if (markType == "__tactics_mvt") {
    emitGemvMark(mark);
  }
  else if (markType == "__tactics_batched_gemm") {
    emitBatchedMatmulMark(mark);
  } 
  else {
    LOG(FATAL) << "Unsupported mark type: " << markType;
  }
}

void AstPrinter::emitAst(isl::ast_node node) {
  if (auto forNode = node.as<isl::ast_node_for>()) {
    emitFor(forNode);
  } else if (auto ifNode = node.as<isl::ast_node_if>()) {
    emitIf(ifNode);
  } else if (auto blockNode = node.as<isl::ast_node_block>()) {
    for (auto child : blockNode.get_children()) {
      emitAst(child);
    }
  } else if (auto mark = node.as<isl::ast_node_mark>()) {
    emitMark(mark);
  } else if (auto userNode = node.as<isl::ast_node_user>()) {
    emitStmt(userNode);
  } else {
    LOG(FATAL) << "NYI " << node << endl;
  }
}

} // namespace

namespace detail {

isl::pw_aff makeAffFromMappedExpr(
    const Halide::Expr& expr,
    const CodegenStatementContext& context) {
  // We only expect this to be called on encountering a free
  // variable. Compound expressions should be emitted as Halide.
  TC_CHECK(expr.as<Halide::Internal::Variable>());
  auto aff = context.makeIslAffFromExpr(expr);
  auto pwaff = isl::pw_aff(aff).pullback(context.iteratorMap());
  return pwaff;
}

isl::space findDomainSpaceById(const CodegenStatementContext& context) {
  for (auto d : isl::UnionAsVector<isl::union_set>(context.scop().domain())) {
    if (d.get_tuple_id() == context.statementId()) {
      return d.get_space();
    }
  }
  TC_CHECK(false) << "could not find domain for " << context.statementId()
                  << " in " << context.scop().domain();
  return isl::space();
}

isl::multi_aff makeMultiAffAccess(
    isl::id tensorId,
    const std::vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context) {
  TC_CHECK_NE(subscripts.size(), 0u)
      << "cannot build subscript aff for a scalar";

  auto domainSpace = findDomainSpaceById(context);
  auto tensorSpace =
      domainSpace.params().add_named_tuple_id_ui(tensorId, subscripts.size());
  auto space = domainSpace.map_from_domain_and_range(tensorSpace);

  auto ma = isl::multi_aff::zero(space);
  for (size_t i = 0; i < subscripts.size(); ++i) {
    ma = ma.set_aff(i, context.makeIslAffFromExpr(subscripts[i]));
  }
  return ma;
}

namespace {
bool is_identifier_or_nonnegative_integer(isl::ast_expr expr) {
  if (auto intExpr = expr.as<isl::ast_expr_int>()) {
    return intExpr.get_val().is_nonneg();
  }
  return !expr.as<isl::ast_expr_id>().is_null();
}
} // namespace

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context,
    const map<string, string>& substitutions) {
  class EmitHalide : public Halide::Internal::IRPrinter {
    using Halide::Internal::IRPrinter::visit;
    void visit(const Halide::Internal::Variable* op) {
      auto pwAff = detail::makeAffFromMappedExpr(Halide::Expr(op), context);
      auto expr = context.build().expr_from(pwAff);
      auto s = expr.to_C_str();
      if (!is_identifier_or_nonnegative_integer(expr)) {
        s = "(" + s + ")";
      }
      context.ss << s;
    }
    void visit(const Halide::Internal::Call* op) {
      if (substitutions.count(op->name)) {
        context.ss << substitutions.at(op->name);
      } else if (
          op->call_type == Halide::Internal::Call::CallType::Halide ||
          op->call_type == Halide::Internal::Call::CallType::Image) {
        detail::emitMappedTensorAccess(op->name, op, op->args, context);
      } else if (op->is_intrinsic(tc2halide::kReductionUpdate)) {
        op->args[0].accept(this);
      } else {
        IRPrinter::visit(op);
      }
    }
    void visit(const Halide::Internal::IntImm* op) {
      context.ss << "(" << halideTypeString(op->type) << ")" << op->value;
    }
    void visit(const Halide::Internal::UIntImm* op) {
      context.ss << "(" << halideTypeString(op->type) << ")" << op->value;
    }
    void visit(const Halide::Internal::FloatImm* op) {
      context.ss << "(" << halideTypeString(op->type) << ")" << op->value;
    }
    void visit(const Halide::Internal::Cast* op) {
      context.ss << "(" << halideTypeString(op->type) << ")";
      context.ss << "(";
      op->value.accept(this);
      context.ss << ")";
    }
    // TODO: handle casts
    const CodegenStatementContext& context;
    const map<string, string>& substitutions;

   public:
    EmitHalide(
        const CodegenStatementContext& ctx,
        const map<string, string>& substitutions)
        : IRPrinter(ctx.ss), context(ctx), substitutions(substitutions) {}
  } printer(context, substitutions);

  e.accept(&printer);
}

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context) {
  map<string, string> substitutions;
  emitHalideExpr(e, context, substitutions);
}

void emitMappedTensorAccess(
    std::string name,
    const Halide::Internal::IRNode* node,
    const vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context) {
  std::cout << __func__ << std::endl;
  // Scalars are not promoted or remapped.
  if (subscripts.empty()) {
    context.ss << name << "[0]";
    return;
  }

  TC_CHECK_EQ(context.scop().halide.accesses.count(node), 1u)
      << "attempting to generate code for tensor " << name
      << " reference not present in Scop" << node;
  auto refId = context.scop().halide.accesses.at(node);

  Scop::PromotionInfo promotionInfo;
  for (auto pi : context.activePromotions()) {
    if (pi.group->referenceIds().count(refId)) {
      TC_CHECK(!promotionInfo.groupId)
          << "reference " << refId
          << " belongs to two groups: " << promotionInfo.groupId << " and "
          << pi.groupId;
      promotionInfo = pi;
    }
  }

  // Not promoted, emitting just the mapped subscript.
  if (!promotionInfo.groupId) {
    context.ss << name;

    for (auto e : subscripts) {
      context.ss << "[";
      emitHalideExpr(e, context);
      context.ss << "]";
    }
    return;
  }

  auto decl = context.scop().promotedDecl(promotionInfo.groupId);
  auto tensorId = decl.tensorId;

  // Here and below in comments: D = domain, O = original tensor, P = promoted
  // tensor, S = partial schedule, A = AST loops;
  // MA = multi_aff, PMA = pw_multi_aff
  auto access =
      makeMultiAffAccess(tensorId, subscripts, context); // MA :: D -> O
  auto promotion = promotionInfo.group->promotion(); // MA :: [S -> O] -> P
  promotion = promotion.set_range_tuple_id(promotionInfo.groupId);
  auto iteratorMap = context.iteratorMap(); // PMA :: A -> D
  auto schedule = isl::map::from(promotionInfo.outerSchedule.intersect_domain(
      context.domain())); // map :: D -> S

  TC_CHECK(schedule.is_single_valued())
      << "expected single-valued schedule, got " << schedule;
  // PMA :: A -> S
  auto astToSchedule = isl::pw_multi_aff(schedule).pullback(iteratorMap);
  // PMA :: A -> O
  auto astToOriginal = isl::pw_multi_aff(access).pullback(iteratorMap);
  // PMA :: A -> [S -> O]
  auto astToScheduledOriginal = astToSchedule.range_product(astToOriginal);
  // PMA :: A -> P
  auto astToPromoted =
      isl::pw_multi_aff(promotion).pullback(astToScheduledOriginal);

  if (decl.kind == Scop::PromotedDecl::Kind::Register) {
    emitRegisterAccess(astToPromoted, context);
  } else {
    emitAccess(astToPromoted, context);
  }
}

} // namespace detail

// TODO: b0,b1,b2 and t0,t1,t2 are actually hardcoded in codegen_cuda
//       bx,by,bz and tx,ty,tz do not work and this is actually scary!!
// TODO: This is terrible and needs to be changed. Funny enough it is already
//       strictly better than the previous implementation...
void emitThreadIdInit(stringstream& ss, const MappedScop& scop) {
  WS ws;
  ss << ws.tab();
  ss << "int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;\n";
  ss << ws.tab();
  ss << "int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;\n";
}

void emitTmpDecl(stringstream& ss, const Scop& scop) {
  for (const auto& kvp : scop.treeSyncUpdateMap) {
    WS ws;
    ss << ws.tab();
    auto updateId = kvp.second;
    auto provide =
        scop.halide.statements.at(updateId).as<Halide::Internal::Provide>();
    ss << halideTypeString(provide->values[0].type()) << " "
       << makeReductionTmpName(updateId, scop) << ";" << endl;
  }
}

void emitPromotedArrayViewsHalide(stringstream& ss, const Scop& scop) {
  for (const auto& p : scop.promotedDecls()) {
    WS ws;
    ss << ws.tab();
    auto viewName = p.first.get_name();
    auto tensorName = p.second.tensorId.get_name();
    Halide::Type t;
    for (auto o : scop.halide.outputs) {
      if (o.name() == tensorName) {
        t = o.type();
      }
    }
    for (auto i : scop.halide.inputs) {
      if (i.name() == tensorName) {
        t = i.type();
      }
    }

    ss << halideTypeString(t) << " " << viewName;
    for (auto s : p.second.sizes) {
      ss << "[" << s << "]";
    }
    ss << ";" << endl;
  }
}

size_t& nAstNodes() {
  static thread_local size_t n = 0;
  return n;
}

string emitTacticsKernel(
    const std::string& specializedName,
    const MappedScop& mscop) {
  // Expecting a schedule with domain root and context first child.
  TC_CHECK(mscop.schedule()->as<ScheduleTreeDomain>());
  TC_CHECK(mscop.schedule()->child({0})->as<ScheduleTreeContext>());

  const auto& scop = mscop.scop();

  // Make a map of the specialized scalar parameter values
  map<string, Halide::Expr> paramValues;
  for (const auto& kvp : scop.parameterValues) {
    paramValues[kvp.first] = kvp.second;
  }

  stringstream ss;
  emitKernelSignature(ss, specializedName, scop);
  emitTensorViews(ss, scop.halide.outputs, paramValues);
  emitTensorViews(ss, scop.halide.inputs, paramValues);
  emitTmpDecl(ss, scop);
  emitPromotedArrayViewsHalide(ss, scop);
  NodeInfoMapType nodeInfoMap;
  auto collect = [&nodeInfoMap](
                     isl::ast_node n, isl::ast_build b) -> isl::ast_node {
    auto collectIteratorMaps =
        [](isl::ast_node node,
           isl::ast_build build,
           NodeInfoMapType* nodeInfoMap) -> isl::ast_node {
      auto user = node.as<isl::ast_node_user>();
      TC_CHECK(user);
      auto expr = user.get_expr().as<isl::ast_expr_op>();
      auto stmtId = expr.get_arg(0).as<isl::ast_expr_id>().get_id();
      auto schedule = build.get_schedule();
      auto scheduleMap = isl::map::from(schedule);

      auto nodeId = isl::id(
          node.get_ctx(),
          std::string(kAstNodeIdPrefix) + std::to_string(nAstNodes()++));
      TC_CHECK_EQ(0u, nodeInfoMap->count(nodeId)) << "entry exists: " << nodeId;

      auto& nodeInfo = (*nodeInfoMap)[nodeId];
      nodeInfo.iteratorMap = isl::pw_multi_aff(scheduleMap.reverse());
      nodeInfo.build = build;
      return node.set_annotation(nodeId);
    };

    return collectIteratorMaps(n, b, &nodeInfoMap);
  };

  auto schedule = detectInSchedule(mscop);
  auto astBuild = isl::ast_build(schedule.get_ctx());
  astBuild = astBuild.set_at_each_domain(collect);

  auto root = ::tc::polyhedral::detail::fromIslSchedule(schedule);

  auto astNode = astBuild.node_from(schedule);

  AstPrinter(CodegenContext(ss, mscop, nodeInfoMap)).emit(astNode);
  ss << "}" << endl;

  return ss.str();
}

string emitTacticsMain(const std::string& specializedName,
		       const MappedScop& mscop)
{
  const Scop& scop = mscop.scop();
  stringstream ss;
  WS ws;

  // Make a map of the specialized scalar parameter values
  map<string, Halide::Expr> paramValues;
  
  for (const auto& kvp : scop.parameterValues)
    paramValues[kvp.first] = kvp.second;

  ss << "int main(int argc, char** argv) {" << std::endl;

  // Parameters
  auto paramsVec = emitParams(scop);

  for(auto& param: paramsVec) {
    ss << ws.tab()
       << "static const "
       << param.second
       << " "
       << param.first
       << " = "
       << paramValues[param.first]
       << ";"
       << std::endl;
  }

  ss << std::endl;
  
  // Declarations
  auto emitDecl = [&](const Halide::OutputImageParam& p) {
		    ss << ws.tab()
		       << "static "
		       << halideTypeString(p.type())
		       << " "
		       << p.name();

		    for (int i = 0; i < p.dimensions(); ++i) {
		      Halide::Expr extent = p.parameter().extent_constraint(i);
		      extent = Halide::Internal::substitute(paramValues, extent);
		      ss << "[" << extent << "]";
		    }

		    ss << ";" << std::endl;
		  };		     
  
  for (auto& o : scop.halide.outputs)
    emitDecl(o);

  ss << std::endl;
  
  for (auto& i : scop.halide.inputs)
    emitDecl(i);

  ss << std::endl;

  // Initializations
  auto emitInit = [&](const Halide::ImageParam& p) {
		    stringstream ssHead;
		    stringstream ssStmt;
		    stringstream tabs;
		    stringstream ssRandExpr;

		    ssStmt << p.name();

		    for (int i = 0; i < p.dimensions(); ++i) {
		      std::string iterName = "i" + std::to_string(i);
		      tabs << ws.tab();
			
		      Halide::Expr extent = p.parameter().extent_constraint(i);

		      ssHead << tabs.str()
			     << "for(size_t "
			     << iterName
			     << " = 0; "
			     << iterName
			     << " < "
			     << extent
			     << "; "
			     << iterName
			     << "++)"
			     << std::endl;

		      ssStmt << "[" << iterName << "]";
		      ssRandExpr << iterName;

		      if(i != p.dimensions()-1)
			ssRandExpr << "+";
		    }

		    ssStmt << " = " << ssRandExpr.str() << ";" << std::endl;

		    tabs << ws.tab();

		    ss << ssHead.str()
		       << tabs.str()
		       << ssStmt.str();
		  };

  for (auto& i : scop.halide.inputs) {
    emitInit(i);
    ss << std::endl;
  }

  ss << std::endl;

  ss << "  " << specializedName << "(";

  auto sigVec = paramsVec +
    emitTypedTensorNames(scop.halide.outputs, makeName) +
    emitTypedTensorNames(scop.halide.inputs, makeName);
 
  for (auto& s : sigVec) {
    string& sname = s.first;

    ss << sname;

    if (s != sigVec.back()) {
      ss << ", ";
    }
  }

  ss << ");" << std::endl;

  // Use the results
  ss << std::endl
     << ws.tab()
     << "float v = 0;"
     << std::endl;

  auto emitUse = [&](const Halide::OutputImageParam& p) {
		   stringstream ssHead;
		   stringstream ssStmt;
		   stringstream tabs;

		   ssStmt << "v += " << p.name();

		   for (int i = 0; i < p.dimensions(); ++i) {
		     std::string iterName = "i" + std::to_string(i);
		     tabs << ws.tab();
			
		     Halide::Expr extent = p.parameter().extent_constraint(i);

		     ssHead << tabs.str()
			    << "for(size_t "
			    << iterName
			    << " = 0; "
			    << iterName
			    << " < "
			    << extent
			    << "; "
			    << iterName
			    << "++)"
			    << std::endl;

		     ssStmt << "[" << iterName << "]";
		   }

		   ssStmt << ";" << std::endl;

		   tabs << ws.tab();

		   ss << ssHead.str()
		      << tabs.str()
		      << ssStmt.str();
		 };

  for (auto& o : scop.halide.outputs) {
    emitUse(o);
    ss << std::endl;
  }

  ss << ws.tab()
     << "printf(\"Sum of all output elements: %f\\n\", v);"
     << std::endl;


  ss << std::endl
     << ws.tab()
     << "return 0;"
     << std::endl;

  ss << "}" << std::endl;

  return ss.str();
}

string emitTacticsEntryPoint(
    const std::string& specializedName,
    const MappedScop& mscop) {
  stringstream ss;

  const auto& scop = mscop.scop();
  auto sigVec = emitParams(scop);
  sigVec = sigVec + emitTypedTensorNames(scop.halide.outputs);
  sigVec = sigVec + emitTypedTensorNames(scop.halide.inputs);

  ss << "void tactics_entrypoint(void** args, size_t num_args) {" << std::endl;

  ss << "  if(num_args != " << sigVec.size() << ") {" << std::endl
     << "    fprintf(stderr, \"Wrong number of arguments:\"" << std::endl
     << "                    \"Expected %zu, but got %zu\\n\", " << std::endl
     << "                    (size_t)" << sigVec.size() << ", num_args);"
     << std::endl
     << "    exit(1);" << std::endl
     << "  }" << std::endl
     << std::endl;

  int i = 0;
  for (auto& s : sigVec) {
    string& sname = s.first;
    string& stype = s.second;

    ss << "  " << stype << " " << sname << " = *((" << stype << "*)args[" << i
       << "]);" << std::endl;

    ++i;
  }

  ss << std::endl;

  ss << "  " << specializedName << "(";

  for (auto& s : sigVec) {
    string& sname = s.first;

    ss << sname;

    if (s != sigVec.back()) {
      ss << ", ";
    }
  }

  ss << ");" << std::endl;

  ss << "}" << std::endl;

  return ss.str();
}

} // namespace tactics
} // namespace polyhedral
} // namespace tc
