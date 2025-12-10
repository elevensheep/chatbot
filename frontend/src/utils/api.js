// API 기본 설정
// 개발 환경: localhost:5000 직접 연결
// 프로덕션 환경: nginx 프록시를 통해 /api로 통일
// nginx의 location /api/는 /api로 시작하는 모든 경로를 매칭하므로 /api만 사용해도 됨
const API_BASE_URL = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:5000/api';

// Refresh Token으로 Access Token 갱신
let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });

  failedQueue = [];
};

async function refreshAccessToken() {
  const refreshToken = localStorage.getItem('refresh_token');

  if (!refreshToken) {
    throw new Error('No refresh token available');
  }

  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ refresh_token: refreshToken })
  });

  if (!response.ok) {
    throw new Error('Token refresh failed');
  }

  const data = await response.json();
  const newAccessToken = data.data?.access_token || data.access_token;

  if (!newAccessToken) {
    throw new Error('No access token in refresh response');
  }

  // 새 토큰 저장
  localStorage.setItem('access_token', newAccessToken);
  localStorage.setItem('authToken', newAccessToken);

  return newAccessToken;
}

// API 호출 유틸리티 함수
export const apiCall = async (endpoint, options = {}) => {
  // endpoint가 /로 시작하므로 그대로 연결 (예: /api + /auth/login = /api/auth/login)
  const url = `${API_BASE_URL}${endpoint}`;

  console.log('apiCall 시작:', url);
  console.log('API_BASE_URL:', API_BASE_URL);

  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const config = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  };

  console.log('fetch 호출 전, config:', config);

  try {
    console.log('fetch 호출 시작...');

    // 타임아웃 설정 (60초) - AI 답변 생성에 시간이 걸리므로 충분한 시간 확보
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.log('fetch 타임아웃 발생!');
      controller.abort();
    }, 60000);

    const response = await fetch(url, { ...config, signal: controller.signal });
    clearTimeout(timeoutId);

    console.log('fetch 응답 받음:', response);
    console.log('응답 상태:', response.status, response.statusText);

    if (!response.ok) {
      // 401 에러이고 Refresh Token이 있으면 갱신 시도
      if (response.status === 401 && localStorage.getItem('refresh_token')) {
        // /auth/refresh 엔드포인트는 재시도하지 않음 (무한 루프 방지)
        if (endpoint !== '/auth/refresh') {
          if (isRefreshing) {
            // 이미 갱신 중이면 큐에 추가
            return new Promise((resolve, reject) => {
              failedQueue.push({ resolve, reject });
            }).then(token => {
              // 새 토큰으로 재시도
              const retryConfig = {
                ...config,
                headers: {
                  ...config.headers,
                  'Authorization': `Bearer ${token}`
                }
              };
              return fetch(url, retryConfig).then(res => res.json());
            });
          }

          isRefreshing = true;

          try {
            const newAccessToken = await refreshAccessToken();
            isRefreshing = false;
            processQueue(null, newAccessToken);

            // 새 토큰으로 원래 요청 재시도
            const retryConfig = {
              ...config,
              headers: {
                ...config.headers,
                'Authorization': `Bearer ${newAccessToken}`
              }
            };

            const retryResponse = await fetch(url, retryConfig);
            if (retryResponse.ok) {
              return await retryResponse.json();
            }
          } catch (refreshError) {
            isRefreshing = false;
            processQueue(refreshError, null);

            // Refresh Token도 만료되었으면 로그아웃
            console.error('토큰 갱신 실패:', refreshError);
            localStorage.clear();
            window.location.href = '/login';
            throw new Error('세션이 만료되었습니다. 다시 로그인해주세요.');
          }
        }
      }

      // 401 에러이지만 Refresh Token이 없거나 갱신 실패한 경우
      if (response.status === 401) {
        localStorage.clear();
        window.location.href = '/login';
        throw new Error('세션이 만료되었습니다. 다시 로그인해주세요.');
      }

      const errorData = await response.json().catch(() => ({}));

      // 백엔드 에러 응답 구조 처리
      // { detail: { error: { message: "...", details: "..." } } } 또는 { detail: "..." } 또는 { message: "..." }
      let errorMessage = `HTTP error! status: ${response.status}`;

      if (errorData.detail) {
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (errorData.detail.error) {
          // error 객체가 있는 경우 message와 details를 조합
          const message = errorData.detail.error.message || '';
          const details = errorData.detail.error.details || '';
          errorMessage = details ? `${message}\n${details}` : message;
        } else if (errorData.detail.message) {
          errorMessage = errorData.detail.message;
        }
      } else if (errorData.message) {
        errorMessage = errorData.message;
      }

      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    console.error('에러 타입:', error.name);
    console.error('에러 메시지:', error.message);
    console.error('에러 전체:', error);
    
    // "Failed to fetch" 오류를 더 명확한 메시지로 변환
    if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
      const friendlyError = new Error(
        '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.\n' +
        `(API URL: ${API_BASE_URL})`
      );
      friendlyError.name = 'NetworkError';
      friendlyError.originalError = error;
      throw friendlyError;
    }
    
    // AbortError (타임아웃) 처리
    if (error.name === 'AbortError') {
      const timeoutError = new Error('요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.');
      timeoutError.name = 'TimeoutError';
      timeoutError.originalError = error;
      throw timeoutError;
    }
    
    throw error;
  }
};

// 인증 관련 API
export const authAPI = {
  // 로그인
  login: async (email, password) => {
    return apiCall('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  // 회원가입
  signup: async (userData) => {
    return apiCall('/auth/signup', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  // 이메일 인증번호 발송
  sendVerificationEmail: async (email) => {
    return apiCall('/auth/send-verification-email', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  // 이메일 인증번호 확인
  verifyEmail: async (email, code) => {
    return apiCall('/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ email, code }),
    });
  },

  // 이메일 토큰으로 검증 (이메일 내 링크 클릭 시 사용)
  verifyWithToken: async (token) => {
    return apiCall('/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ token }),
    });
  },

  // 로그아웃
  logout: async () => {
    return apiCall('/auth/logout', {
      method: 'POST',
    });
  },

  // 토큰 검증
  verifyToken: async (token) => {
    return apiCall('/auth/verify', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });
  },

  // 비밀번호 재설정 요청
  requestPasswordReset: async (email) => {
    return apiCall('/auth/password-reset/request', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  // 비밀번호 재설정 확인
  confirmPasswordReset: async (email, token, newPassword) => {
    return apiCall('/auth/password-reset/confirm', {
      method: 'POST',
      body: JSON.stringify({
        email,
        token,
        new_password: newPassword
      }),
    });
  },
};

// 대화방 목록 조회
export const getConversations = async () => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall('/conversations', {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  // 백엔드 응답 구조: { success: true, data: { conversations: [...], total: N } }
  return response.data?.conversations || response.conversations || [];
};

// 새 대화방 생성
export const createConversation = async (title) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall('/conversations', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });
  // 백엔드 응답 구조: { success: true, data: {...} }
  return response.data || response;
};

// 특정 대화방 메시지 조회
export const getMessages = async (conversationId) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall(`/conversations/${conversationId}/messages`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  // 백엔드 응답 구조: { success: true, data: { messages: [...], total: N } }
  return response.data || response;
};

// 메시지 전송
export const sendMessage = async (conversationId, query, k = 5, includeSources = true) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall('/conversations/chat', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ conversation_id: conversationId, query, k, include_sources: includeSources }),
  });
  // 백엔드 응답 구조: { success: true, data: {...} }
  return response.data || response;
};

// 대화방 삭제
export const deleteConversation = async (conversationId) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall(`/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  // 백엔드 응답 구조: { success: true, message: "..." }
  return response;
};

export { API_BASE_URL };