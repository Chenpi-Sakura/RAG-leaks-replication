import time
from data_manager import DataManager
from llm_service import LocalTransformersLLM, APIBasedLLM
from rag_system import SimpleRAG
from attack_core import DCMIA  # DC-MIA = Difficulty-Calibrated MIA (RAG-leaks)
from evaluator import Evaluator

def main():
    print("=== 开始运行 DC-MIA 评测脚本 ===")
    
    # 1. 准备数据
    dm = DataManager(data_dir="./data")
    # 为了演示，生成极少量数据跑通流程。实际跑批时使用 12000。
    # 临时调大到 12000 以验证 phase 2 不会因为 reference_pool 为空而崩
    # 原值 3000 是 "跑通流程用"，data_manager 切分逻辑要求 >=11100
    raw_data = dm.load_or_generate_dummy_data(total_samples=12000)
    splits = dm.split_data(raw_data)
    
    print(f"数据划分完成:")
    print(f"  - 目标知识库: {len(splits['target_kb'])}")
    print(f"  - 辅助集: {len(splits['aux_data'])}")
    print(f"  - 测试集: {len(splits['eval_data'])}")
    print(f"  - 影子 RAG 采样池: {len(splits['reference_pool'])}")

    # 2. 初始化大模型服务 (这里默认用 Mock 模型跑通代码骨架)
    # 切换到真实本地大模型时，只需: llm = LocalTransformersLLM("Qwen/Qwen2-7B-Instruct", use_unsloth=True)
    llm = LocalTransformersLLM("mock-model", use_unsloth=False)
    
    # 3. 初始化 Target RAG (被攻击的靶机)
    print("\n[Init] 正在构建 Target RAG 知识库...")
    target_rag = SimpleRAG(llm_service=llm)
    target_rag.build_index(splits['target_kb'])
    
    # 4. 初始化攻击引擎
    attacker = DCMIA(llm_service=llm, data_pool=splits['reference_pool'])  # DC-MIA
    
    # --- 攻击开始 ---
    print("\n[Attack Phase 1] 寻找高相似度阈值...")
    tau_1 = attacker.phase_1_find_threshold(target_rag, splits['aux_data'])
    
    y_true = []
    y_scores = []

    # 这里的 member 判定仅仅是为了算指标时传真实标签
    # 实际上我们评估集是前1000为member, 后1000为non-member，具体可根据 id 在 target_kb 里查
    target_ids = {s['id'] for s in splits['target_kb']}

    print(f"\n[Attack Phase 2] 开始对测试集进行难度校准成员推断...")
    # 使用完整 eval_data (2000 条: 1000 member + 1000 non-member) 保证 AUC 可计算
    # 修复 Bug P1: 之前 [:10] 切片全是 member，单类别 AUC=nan
    test_samples = splits['eval_data']
    n_member = sum(1 for s in test_samples if s['id'] in target_ids)
    print(f"  - 评测集样本数: {len(test_samples)} (member={n_member}, non-member={len(test_samples) - n_member})")
    
    start_time = time.time()
    for i, sample in enumerate(test_samples):
        # 执行完整的 DC-MIA 攻击 (如果相似度高直接返回极大值，否则跑 16 个影子 RAG 算似然比)
        # 注意: 论文中默认 m=8, 即 8 对 in/out RAGs。这里为了演示缩短到 m=2
        score = attacker.attack(target_rag, sample, tau_1, m=2) 
        
        y_scores.append(score)
        y_true.append(1 if sample['id'] in target_ids else 0)
        
        print(f"  - 样本 {sample['id']} 攻击完成, 校准得分 (Likelihood Ratio): {score:.4f}")

    print(f"\n[Done] 攻击完成，耗时 {time.time() - start_time:.2f} 秒")
    
    # 5. 评测结果
    Evaluator.calculate_metrics(y_true, y_scores)

if __name__ == "__main__":
    main()
