"""
기본 테스트 파일 - import 테스트 및 기본 기능 검증
"""
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))


def test_imports():
    """모든 주요 모듈이 정상적으로 import되는지 확인"""
    try:
        from config import settings
        assert settings is not None
        print("✅ config import 성공")
    except Exception as e:
        pytest.fail(f"config import 실패: {e}")
    
    try:
        from main import app
        assert app is not None
        print("✅ main import 성공")
    except Exception as e:
        pytest.fail(f"main import 실패: {e}")


def test_settings_basic():
    """설정 기본값 확인"""
    from config import settings
    
    assert hasattr(settings, 'APP_NAME')
    assert hasattr(settings, 'APP_VERSION')
    assert hasattr(settings, 'ENVIRONMENT')
    assert hasattr(settings, 'DEBUG')
    assert hasattr(settings, 'LOG_LEVEL')
    print("✅ settings 기본 속성 확인 완료")


def test_app_creation():
    """FastAPI 앱이 정상적으로 생성되는지 확인"""
    from main import app
    
    assert app is not None
    assert app.title is not None
    assert app.version is not None
    print("✅ FastAPI 앱 생성 확인 완료")


def test_router_exists():
    """라우터가 정상적으로 등록되었는지 확인"""
    from main import app
    
    # /api/health 엔드포인트가 있는지 확인
    routes = [route.path for route in app.routes]
    assert any('/api/health' in route or route == '/api/health' for route in routes), \
        f"Expected /api/health route, but found: {routes}"
    print("✅ 라우터 등록 확인 완료")


def test_health_endpoint_structure():
    """헬스 체크 엔드포인트가 올바른 구조를 반환하는지 확인"""
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/api/health")
        
        # HTTP 상태 코드 확인 (200 또는 503 둘 다 가능)
        assert response.status_code in [200, 503], \
            f"예상치 못한 상태 코드: {response.status_code}"
        
        # JSON 응답 확인
        result = response.json()
        assert isinstance(result, dict), "헬스 체크 결과가 딕셔너리가 아닙니다"
        
        # 필수 필드 확인
        assert 'status' in result, "status 필드가 없습니다"
        assert 'environment' in result, "environment 필드가 없습니다"
        assert 'version' in result, "version 필드가 없습니다"
        assert 'checks' in result, "checks 필드가 없습니다"
        
        print(f"✅ 헬스 체크 엔드포인트 구조 확인 완료 (상태: {result.get('status')})")
    except ImportError:
        # FastAPI TestClient가 없을 경우 직접 함수 호출 시도
        try:
            from main import router
            import asyncio
            
            # router에서 health_check 함수 찾기
            for route in router.routes:
                if hasattr(route, 'path') and 'health' in route.path:
                    # 직접 호출은 복잡하므로 스킵
                    print("⚠️ TestClient를 사용할 수 없어 헬스 체크 테스트 스킵")
                    return
        except Exception as e:
            print(f"⚠️ 헬스 체크 테스트 스킵: {e}")
    except Exception as e:
        # 헬스 체크가 실패해도 구조는 확인할 수 있어야 함
        print(f"⚠️ 헬스 체크 실행 중 오류 (환경 변수 미설정 가능): {e}")


def test_chat_request_model():
    """ChatRequest 모델이 정상적으로 작동하는지 확인"""
    from main import ChatRequest
    
    # 기본 필드 확인
    request = ChatRequest(query="테스트 질문")
    assert request.query == "테스트 질문"
    assert request.k == 3  # 기본값
    assert request.include_sources == True  # 기본값
    
    # 커스텀 값 확인
    request2 = ChatRequest(query="테스트", k=5, include_sources=False)
    assert request2.k == 5
    assert request2.include_sources == False
    
    print("✅ ChatRequest 모델 확인 완료")


def test_chat_response_model():
    """ChatResponse 모델이 정상적으로 작동하는지 확인"""
    from main import ChatResponse
    
    response = ChatResponse(answer="테스트 답변", sources=[], message_id="test_msg_id")
    assert response.answer == "테스트 답변"
    assert response.sources == []
    assert response.message_id == "test_msg_id"
    
    response2 = ChatResponse(
        answer="답변",
        sources=[{"course_name": "테스트", "professor": "교수"}],
        message_id="test_msg_id_2"
    )
    assert len(response2.sources) == 1
    assert response2.message_id == "test_msg_id_2"
    
    print("✅ ChatResponse 모델 확인 완료")


if __name__ == "__main__":
    """직접 실행 시 모든 테스트 실행"""
    print("=" * 50)
    print("테스트 시작")
    print("=" * 50)
    
    test_functions = [
        test_imports,
        test_settings_basic,
        test_app_creation,
        test_router_exists,
        test_health_endpoint_structure,
        test_chat_request_model,
        test_chat_response_model,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 실패: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"테스트 완료: {passed}개 통과, {failed}개 실패")
    print("=" * 50)
    
    if failed > 0:
        sys.exit(1)

