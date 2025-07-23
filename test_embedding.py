import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.encoders import UnifiedTextEncoder

def test_multilingual_embedding():
    """멀티링구얼 임베딩 모델 테스트"""
    print("=== 멀티링구얼 임베딩 모델 테스트 ===\n")
    
    # 모델 초기화
    encoder = UnifiedTextEncoder(
        embed_dim=768,
        pretrained_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 테스트 데이터 (한국어, 영어, 일본어)
    test_lyrics = [
        "[verse 1:piano] 사랑하는 사람아 너를 위한 노래",
        "[chorus:guitar] I love you more than words can say", 
        "[bridge:violin] 君を愛している 永遠に",
        "[verse 2:drums] 우리 함께 걸었던 그 길을 기억해"
    ]
    
    test_styles = [
        "romantic ballad, slow tempo",
        "energetic pop rock, upbeat", 
        "classical crossover, emotional",
        "korean folk, traditional"
    ]
    
    print("테스트 입력:")
    for i, (lyric, style) in enumerate(zip(test_lyrics, test_styles)):
        print(f"{i+1}. 가사: {lyric}")
        print(f"   스타일: {style}\n")
    
    try:
        # 모델 실행
        print("임베딩 생성 중...")
        
        with torch.no_grad():
            embeddings = encoder(lyrics=test_lyrics, style=test_styles)
        
        print(f"SUCCESS! 임베딩 shape: {embeddings.shape}")
        print(f"임베딩 차원: {embeddings.size(-1)}")
        print(f"배치 크기: {embeddings.size(0)}")
        
        # 각 임베딩의 norm 확인
        norms = torch.norm(embeddings, dim=-1)
        print(f"\n임베딩 벡터 크기:")
        for i, norm in enumerate(norms):
            print(f"  샘플 {i+1}: {norm:.4f}")
        
        # 유사도 테스트
        print(f"\n유사도 매트릭스 (코사인 유사도):")
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        
        for i in range(len(test_lyrics)):
            row = " ".join([f"{similarity[i,j]:.3f}" for j in range(len(test_lyrics))])
            print(f"  샘플 {i+1}: [{row}]")
        
        # 개별 컴포넌트 테스트
        print(f"\n=== 개별 컴포넌트 테스트 ===")
        
        # 가사만
        lyrics_only = encoder(lyrics=test_lyrics, style=None)
        print(f"가사만: {lyrics_only.shape}")
        
        # 스타일만  
        style_only = encoder(lyrics=None, style=test_styles)
        print(f"스타일만: {style_only.shape}")
        
        print(f"\nALL TESTS PASSED!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_vocab_building():
    """동적 vocab 구축 테스트"""
    print("\n=== 동적 Vocab 구축 테스트 ===")
    
    encoder = UnifiedTextEncoder()
    
    # 샘플 데이터로 vocab 구축
    sample_lyrics = [
        "[verse 1:piano] 한국어 가사 테스트",
        "[chorus:guitar:energetic] English lyrics test",
        "[bridge:violin] 混合语言测试", 
        "[outro:drums:slow] Final verse"
    ]
    
    sample_styles = [
        "romantic ballad, emotional",
        "rock, energetic, powerful",
        "classical, peaceful, serene"
    ]
    
    print("Vocab 구축 전:")
    print(f"Section vocab: {len(encoder.section_vocab)}")
    print(f"Style vocab: {len(encoder.style_vocab)}")
    
    encoder.build_vocab_from_data(sample_lyrics, sample_styles)
    
    print(f"\nVocab 구축 후:")
    stats = encoder.get_vocab_stats()
    print(f"Section vocab: {stats['section_vocab_size']}")
    print(f"Style vocab: {stats['style_vocab_size']}")
    print(f"Sections found: {stats['sections']}")
    print(f"Styles found: {stats['styles']}")

if __name__ == "__main__":
    test_multilingual_embedding()
    test_vocab_building()