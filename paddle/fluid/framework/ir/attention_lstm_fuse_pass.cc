// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/attention_lstm_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace ir {

AttentionLSTMFusePass::AttentionLSTMFusePass() {
  AddOpCompat(OpCompat("while"))
      .AddInput("X")  // A set of variables, unconstrained
      .End()
      .AddInput("Condition")  // An scalar
      .IsTensor()
      .End()
      .AddOutput("Out")  // A set of variables, unconstrained
      .End()
      .AddOutput("StepScopes")  // A vector of local scope, unconstrained
      .End()
      .AddAttr("sub_block")
      .IsType<framework::BlockDesc*>()
      .End();

  AddOpCompat(OpCompat("fill_constant"))
      .AddInput("ValueTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensorList")  // vector<phi::DenseTensor<int>>
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("dtype")
      .IsNumGE(0)
      .IsNumLE(25)
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("value")
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("sequence_expand"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("ref_level")
      .IsNumGE(-1)
      .End();
}
struct Param {
  std::string X = "concat_0.tmp_0";
  std::string C0 = "cell_init";
  std::string H0 = "hidden_init";
  std::string AttentionWeight = "attention_fc.w_0";
  std::string AttentionBias = "attention_fc.b_0";
  std::string AttentionScalar = "attention_output.w_0";
  std::string AttentionScalarBias = "attention_output.b_0";
  std::string LSTMWeight = "attention_w.new";
  std::string LSTMBias = "attention_b.new";
  std::string Hidden = "array_to_lod_tensor_0.tmp_0";
  std::string Cell = "at.cell.new";
  std::string AttentionedX = "at.x.new";
  std::string AttentionFCOut = "at.fc.new";
  std::string LSTMX = "at.lstmx.new";
  std::string LSTMOUT = "at.lstmout.new";
};

void PrepareParameters(Graph* graph, const Param& param, ir::Node* lstm_op);

void AttentionLSTMFusePass::FindWhileOp(Graph* graph) const {
  GraphPatternDetector gpd;
  std::unordered_set<int> fused_external_ops(
      {35, 36, 37, 38, 43, 44, 49, 45, 46, 47, 41, 42, 53, 54, 48,
       57, 55, 56, 52, 74, 80, 77, 78, 79, 50, 77, 39, 40, 51});

  gpd.mutable_pattern()->NewNode(
      [&](Node* n) { return fused_external_ops.count(n->id()); }, "while");

  auto& marked_nodes = GetMarkedNodes(graph);

  auto handle = [&](const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    auto* while_pat_node = gpd.pattern().RetrieveNode("while");
    auto* while_node = subgraph.at(while_pat_node);
    marked_nodes.insert(while_node);
  };
  gpd(graph, handle);

  Param param;
  // Add AttentionLSTM node
  OpDesc op_desc;
  op_desc.SetType("attention_lstm");

#define OP_SET_IN(x) op_desc.SetInput(#x, {param.x});
#define OP_SET_OUT(x) op_desc.SetOutput(#x, {param.x});
  OP_SET_IN(X);
  OP_SET_IN(C0);
  OP_SET_IN(H0);
  OP_SET_IN(AttentionWeight);
  OP_SET_IN(AttentionBias);
  OP_SET_IN(AttentionScalar);
  OP_SET_IN(AttentionScalarBias);
  OP_SET_IN(LSTMWeight);
  OP_SET_IN(LSTMBias);

  OP_SET_OUT(Hidden);
  OP_SET_OUT(Cell);
  OP_SET_OUT(AttentionedX);
  OP_SET_OUT(AttentionFCOut);
  OP_SET_OUT(LSTMX);
  OP_SET_OUT(LSTMOUT);
#undef OP_SET_IN
#undef OP_SET_OUT

  auto* X = graph->RetrieveNode(34);
  auto* LSTMOUT = graph->RetrieveNode(81);
  auto* cell_init = graph->RetrieveNode(6);
  auto* hidden_init = graph->RetrieveNode(8);

  auto* lstm_op = graph->CreateOpNode(&op_desc);
  PrepareParameters(graph, param, lstm_op);

  IR_NODE_LINK_TO(X, lstm_op);
  IR_NODE_LINK_TO(cell_init, lstm_op);
  IR_NODE_LINK_TO(hidden_init, lstm_op);
  IR_NODE_LINK_TO(lstm_op, LSTMOUT);

  GraphSafeRemoveNodes(graph, marked_nodes);
}

#define CHECK_P1(x)        \
  PADDLE_ENFORCE_NOT_NULL( \
      x, platform::errors::NotFound("%s is a null pointer.", #x))
#define CHECK_P2(x0, x1) \
  CHECK_P1(x0);          \
  CHECK_P1(x1);
#define CHECK_P3(x0, x1, x2) \
  CHECK_P2(x0, x1);          \
  CHECK_P1(x2);
#define CHECK_P4(x0, x1, x2, x3) \
  CHECK_P3(x0, x1, x2);          \
  CHECK_P1(x3);
#define CHECK_P5(x0, x1, x2, x3, x4) \
  CHECK_P4(x0, x1, x2, x3);          \
  CHECK_P1(x4);

void PrepareLSTMWeight(const phi::DenseTensor& W_forget_w0,
                       const phi::DenseTensor& W_forget_w1,
                       const phi::DenseTensor& W_input_w0,
                       const phi::DenseTensor& W_input_w1,
                       const phi::DenseTensor& W_output_w0,
                       const phi::DenseTensor& W_output_w1,
                       const phi::DenseTensor& W_cell_w0,
                       const phi::DenseTensor& W_cell_w1,
                       phi::DenseTensor* out);

void PrepareLSTMBias(const phi::DenseTensor& B_forget,
                     const phi::DenseTensor& B_input,
                     const phi::DenseTensor& B_output,
                     const phi::DenseTensor& B_cell,
                     phi::DenseTensor* out);

void PrepareParameters(Graph* graph, const Param& param, ir::Node* lstm_op) {
  // Check parameters
  PADDLE_ENFORCE_EQ(graph->Has(kParamScopeAttr),
                    true,
                    platform::errors::InvalidArgument(
                        "Graph have no attribute: kParamScopeAttr."));
  auto& scope = graph->Get<Scope>(kParamScopeAttr);

  // Create new parameters.
  // AddInput
  scope.Var(param.LSTMWeight)->GetMutable<phi::DenseTensor>();
  scope.Var(param.LSTMBias)->GetMutable<phi::DenseTensor>();
// AddOutput
#define IR_NODE(x)                                 \
  VarDesc key_##x(param.x);                        \
  key_##x.SetPersistable(false);                   \
  auto* node_##x = graph->CreateVarNode(&key_##x); \
  IR_NODE_LINK_TO(lstm_op, node_##x);

  IR_NODE(Hidden);
  IR_NODE(Cell);
  IR_NODE(AttentionedX);
  IR_NODE(AttentionFCOut);
  IR_NODE(LSTMX);
  IR_NODE(LSTMOUT);
#undef IR_NODE

#define GATE_W(name__)                                                      \
  auto* W_##name__##_w0 = scope.FindVar(#name__ ".w_0");                    \
  auto* W_##name__##_w1 = scope.FindVar(#name__ ".w_1");                    \
  auto* W_##name__##_b0 = scope.FindVar(#name__ ".b_0");                    \
  CHECK_P3(W_##name__##_w0, W_##name__##_w1, W_##name__##_b0);              \
  VLOG(4) << #name__ "_w0"                                                  \
          << " shape: " << W_##name__##_w0->Get<phi::DenseTensor>().dims(); \
  VLOG(4) << #name__ "_w1"                                                  \
          << " shape: " << W_##name__##_w1->Get<phi::DenseTensor>().dims(); \
  VLOG(4) << #name__ "_b0"                                                  \
          << " shape: " << W_##name__##_b0->Get<phi::DenseTensor>().dims(); \
  auto& W_##name__##_w0_t = W_##name__##_w0->Get<phi::DenseTensor>();       \
  auto& W_##name__##_w1_t = W_##name__##_w1->Get<phi::DenseTensor>();       \
  auto& W_##name__##_b0_t = W_##name__##_b0->Get<phi::DenseTensor>();

  GATE_W(forget);
  GATE_W(input);
  GATE_W(output);
  GATE_W(c);
#undef GATE_W

  auto* attention_fc_w = scope.FindVar("attention_fc.w_0");
  auto* attention_fc_b = scope.FindVar("attention_fc.b_0");
  auto* attention_output_w = scope.FindVar("attention_output.w_0");
  auto* attention_output_b = scope.FindVar("attention_output.b_0");
  CHECK_P4(
      attention_fc_w, attention_fc_b, attention_output_w, attention_output_b);

  auto* lstm_weight = scope.Var(param.LSTMWeight);
  auto* lstm_weight_t = lstm_weight->GetMutable<phi::DenseTensor>();
  auto* lstm_bias = scope.Var(param.LSTMBias);
  auto* lstm_bias_t = lstm_bias->GetMutable<phi::DenseTensor>();

  // reshape attention_bias
  auto* attention_bias_t =
      scope.FindVar(param.AttentionBias)->GetMutable<phi::DenseTensor>();
  PADDLE_ENFORCE_EQ(
      attention_bias_t->dims().size(),
      1,
      platform::errors::InvalidArgument(
          "phi::DenseTensor attention bias dimension size(%d) must be 1.",
          attention_bias_t->dims().size()));
  attention_bias_t->Resize(phi::make_ddim({1, attention_bias_t->dims()[0]}));

  auto* attention_scalar_bias_t =
      scope.FindVar(param.AttentionScalarBias)->GetMutable<phi::DenseTensor>();
  attention_scalar_bias_t->Resize(
      phi::make_ddim({1, attention_scalar_bias_t->dims()[0]}));

  PrepareLSTMWeight(W_forget_w0_t,
                    W_forget_w1_t,
                    W_input_w0_t,
                    W_input_w1_t,
                    W_output_w0_t,
                    W_output_w1_t,
                    W_c_w0_t,
                    W_c_w1_t,
                    lstm_weight_t);
  PrepareLSTMBias(
      W_forget_b0_t, W_input_b0_t, W_output_b0_t, W_c_b0_t, lstm_bias_t);
}

// Prepare parameters
void PrepareLSTMWeight(const phi::DenseTensor& W_forget_w0,
                       const phi::DenseTensor& W_forget_w1,
                       const phi::DenseTensor& W_input_w0,
                       const phi::DenseTensor& W_input_w1,
                       const phi::DenseTensor& W_output_w0,
                       const phi::DenseTensor& W_output_w1,
                       const phi::DenseTensor& W_cell_w0,
                       const phi::DenseTensor& W_cell_w1,
                       phi::DenseTensor* out) {
  int D = static_cast<int>(W_forget_w0.dims()[0]);
  int M = static_cast<int>(W_forget_w1.dims()[0]);
  out->Resize(phi::make_ddim({D + M, 4 * D}));
  VLOG(3) << "LSTMWeight resized to " << out->dims();

  float* out_data = out->mutable_data<float>(platform::CPUPlace());
  std::array<const float*, 4> tensors{W_forget_w0.data<float>(),
                                      W_input_w0.data<float>(),
                                      W_output_w0.data<float>(),
                                      W_cell_w0.data<float>()};
  std::array<const float*, 4> tensors1{W_forget_w1.data<float>(),
                                       W_input_w1.data<float>(),
                                       W_output_w1.data<float>(),
                                       W_cell_w1.data<float>()};

  for (int row = 0; row < D; row++) {
    for (int col = 0; col < 4; col++) {
      float* dst = out_data + 4 * D * row + D * col;
      const float* src = tensors[col] + D * row;
      memcpy(dst, src, D * sizeof(float));
    }
  }

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < 4; col++) {
      float* dst = out_data + 4 * D * (D + row) + D * col;
      const float* src = tensors1[col] + D * row;
      memcpy(dst, src, D * sizeof(float));
    }
  }
}

void PrepareLSTMBias(const phi::DenseTensor& B_forget,
                     const phi::DenseTensor& B_input,
                     const phi::DenseTensor& B_output,
                     const phi::DenseTensor& B_cell,
                     phi::DenseTensor* out) {
  std::array<const float*, 4> tensors{B_forget.data<float>(),
                                      B_input.data<float>(),
                                      B_output.data<float>(),
                                      B_cell.data<float>()};

  PADDLE_ENFORCE_EQ(
      B_forget.dims().size(),
      1,
      platform::errors::InvalidArgument(
          "phi::DenseTensor B forget dimension size(%d) must be 1.",
          B_forget.dims().size()));
  int D = static_cast<int>(B_forget.dims()[0]);
  out->Resize(phi::make_ddim({1, 4 * D}));
  auto* out_data = out->mutable_data<float>(platform::CPUPlace());
  for (size_t i = 0; i < tensors.size(); i++) {
    memcpy(out_data + D * i, tensors[i], D * sizeof(float));
  }
}

// Parameters

void AttentionLSTMFusePass::ApplyImpl(ir::Graph* graph) const {
  PDPattern external_pattern, subblock_pattern;

  // Use the following variables to tell whether this model is RNN1.
  // This fuse can only works on the RNN1 model.
  std::unordered_set<std::string> specified_vars({"data_lod_attention",
                                                  "cell_init",
                                                  "hidden_init",
                                                  "data",
                                                  "week",
                                                  "minute"});
  size_t count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsVar() && specified_vars.count(node->Name())) {
      ++count;
    }
  }
  if (count < specified_vars.size()) {
    return;
  }

  // Continue to fuse.
  FindWhileOp(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(attention_lstm_fuse_pass,
              paddle::framework::ir::AttentionLSTMFusePass);
