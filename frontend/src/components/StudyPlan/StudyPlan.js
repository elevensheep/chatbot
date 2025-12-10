import React, { useState } from 'react';
import './StudyPlan.css';
import StudyPlanSave from '../StudyPlanSave/StudyPlanSave';

export default function StudyPlan() {
  const [formData, setFormData] = useState({
    // 교과목정보
    professor: '',
    studentCount: '',
    courseCode: '',
    department: '',
    major: '',
    credits: '',
    hours: '',
    email: '',
    
    // 교과목역량 헤더
    totalStudents: '',
    capacity: '',
    
    // 상세 정보
    courseOverview: '',
    objectives: '',
    mainTextbook: '',
    subTextbook: '',
    references: ''
  });

  const [savedPlans, setSavedPlans] = useState([]); // 여러 수업계획서 저장
  const [selectedPlanId, setSelectedPlanId] = useState(null);

  // 주차별 수업계획 (1~15주차)
  const makeDefaultWeeklyPlans = () => (
    Array.from({ length: 15 }, (_, idx) => ({
      week: idx + 1,
      topic: '',
      method: '',
      strategy: ''
    }))
  );

  const [weeklyPlans, setWeeklyPlans] = useState(makeDefaultWeeklyPlans());
  
  // 교과목역량 분야 배열
  const [competencyFields, setCompetencyFields] = useState([
    {
      id: 1,
      fieldName: '',
      learningCompetency: '',
      competencyFactor: '',
      learningHours: '',
      classificationRatio: '',
      teachingMethod: '',
      evaluation: '',
      evaluationCount: ''
    },
    {
      id: 2,
      fieldName: '',
      learningCompetency: '',
      competencyFactor: '',
      learningHours: '',
      classificationRatio: '',
      teachingMethod: '',
      evaluation: '',
      evaluationCount: ''
    }
  ]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // 교과목역량 필드 변경 핸들러
  const handleCompetencyChange = (id, field, value) => {
    setCompetencyFields(prev => 
      prev.map(item => 
        item.id === id ? { ...item, [field]: value } : item
      )
    );
  };

  // 주차별 수업계획 변경 핸들러
  const handleWeeklyPlanChange = (week, field, value) => {
    setWeeklyPlans(prev =>
      prev.map(item =>
        item.week === week ? { ...item, [field]: value } : item
      )
    );
  };

  // 교과목역량 분야 추가
  const addCompetencyField = () => {
    const newId = Math.max(...competencyFields.map(f => f.id)) + 1;
    setCompetencyFields(prev => [...prev, {
      id: newId,
      fieldName: '',
      learningCompetency: '',
      competencyFactor: '',
      learningHours: '',
      classificationRatio: '',
      teachingMethod: '',
      evaluation: '',
      evaluationCount: ''
    }]);
  };

  // 교과목역량 분야 삭제
  const removeCompetencyField = (id) => {
    if (competencyFields.length > 1) { // 최소 1개는 유지
      setCompetencyFields(prev => prev.filter(item => item.id !== id));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // 교과목 제목이 없으면 저장하지 않음
    if (!formData.courseCode.trim()) {
      alert('교과목 제목을 입력해주세요.');
      return;
    }
    
    // 새로운 수업계획서 생성
    const newPlan = {
      id: Date.now(), // 유니크 ID
      title: formData.courseCode,
      data: {...formData},
      competencyFields: [...competencyFields],
      weeklyPlans: [...weeklyPlans],
      createdAt: new Date().toLocaleString()
    };
    
    // 저장된 계획서 목록에 추가
    setSavedPlans(prev => [...prev, newPlan]);
    setSelectedPlanId(newPlan.id);
    
    console.log('수업계획서 저장됨:', newPlan);
    alert(`"${formData.courseCode}" 수업계획서가 저장되었습니다.`);
  };

  // 저장된 계획서 선택
  const handleSelectPlan = (planId) => {
    const selectedPlan = savedPlans.find(plan => plan.id === planId);
    if (selectedPlan) {
      setSelectedPlanId(planId);
      // 폼 데이터를 선택된 계획서로 업데이트
      setFormData(selectedPlan.data);
      // 교과목역량 필드들도 업데이트
      if (selectedPlan.competencyFields) {
        setCompetencyFields(selectedPlan.competencyFields);
      }
      // 주차별 수업계획도 업데이트 (없으면 기본값)
      if (selectedPlan.weeklyPlans && selectedPlan.weeklyPlans.length === 15) {
        setWeeklyPlans(selectedPlan.weeklyPlans);
      } else {
        setWeeklyPlans(makeDefaultWeeklyPlans());
      }
    }
  };

  // 저장된 계획서 삭제
  const handleDeletePlan = (planId) => {
    if (window.confirm('이 수업계획서를 삭제하시겠습니까?')) {
      setSavedPlans(prev => prev.filter(plan => plan.id !== planId));
      if (selectedPlanId === planId) {
        setSelectedPlanId(null);
      }
    }
  };

  // 새 계획서 시작
  const handleNewPlan = () => {
    setFormData({
      professor: '',
      studentCount: '',
      courseCode: '',
      department: '',
      major: '',
      credits: '',
      hours: '',
      email: '',
      totalStudents: '',
      capacity: '',
      courseOverview: '',
      objectives: '',
      mainTextbook: '',
      subTextbook: '',
      references: ''
    });
    setCompetencyFields([
      {
        id: 1,
        learningCompetency: '',
        competencyFactor: '',
        learningHours: '',
        classificationRatio: '',
        teachingMethod: '',
        evaluation: '',
        evaluationCount: ''
      },
      {
        id: 2,
        learningCompetency: '',
        competencyFactor: '',
        learningHours: '',
        classificationRatio: '',
        teachingMethod: '',
        evaluation: '',
        evaluationCount: ''
      }
    ]);
    setSelectedPlanId(null);
    setWeeklyPlans(makeDefaultWeeklyPlans());
  };

  return (
    <div className="study-plan-main-container">
      {/* 왼쪽: 입력 폼 */}
      <div className="study-plan-left">
        <div className="study-plan-container">
          <div className="study-plan-header">
            <h1>2025학년도 2학기 수업계획서</h1>
          </div>

          <form onSubmit={handleSubmit} className="study-plan-form">
        {/* 교과목정보 섹션 */}
        <div className="section-title">교과목정보</div>
        <table className="info-table">
          <tbody>
            <tr>
              <td className="label-cell">담당교수</td>
              <td className="input-cell">
                <input
                  type="text"
                  name="professor"
                  value={formData.professor}
                  onChange={handleInputChange}
                  placeholder="예) 홍길동"
                />
              </td>
              <td className="label-cell">교과목</td>
              <td className="input-cell">
                <input
                  type="text"
                  name="courseCode"
                  value={formData.courseCode}
                  onChange={handleInputChange}
                  placeholder="예) [17] 프로그래밍기초"
                />
              </td>
              <td className="label-cell">이수구분</td>
              <td className="input-cell">
                <input
                  type="text"
                  name="department"
                  value={formData.department}
                  onChange={handleInputChange}
                  placeholder="전공"
                />
              </td>
            </tr>
            <tr>
              <td className="label-cell">시간/학점</td>
              <td className="input-cell">
                <input
                  type="text"
                  name="credits"
                  value={formData.credits}
                  onChange={handleInputChange}
                  placeholder="예) 3/3"
                />
              </td>
              <td className="label-cell">이론/실습</td>
              <td className="input-cell">
                <input
                  type="text"
                  name="hours"
                  value={formData.hours}
                  onChange={handleInputChange}
                  placeholder="예) 2/1"
                />
              </td>
              <td className="label-cell"></td>
              <td className="input-cell"></td>
            </tr>
            <tr>
              <td className="label-cell">연락처</td>
              <td className="input-cell">
                <input
                  type="text"
                  name="contact"
                  value={formData.contact}
                  onChange={handleInputChange}
                  placeholder="예) 010-1234-5678"
                />
              </td>
              <td className="label-cell">E-Mail</td>
              <td className="input-cell" colSpan="3">
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  placeholder="예) professor@university.ac.kr"
                />
              </td>
            </tr>
          </tbody>
        </table>

        {/* 교과목역량 섹션 */}
        <div className="section-title">교과목역량</div>
        <table className="capacity-table">
          <thead>
            <tr>
              <td className="label-cell gray-label">수강인수(시간)</td>
              <td className="number-cell">
                <input
                  type="number"
                  name="totalStudents"
                  value={formData.totalStudents}
                  onChange={handleInputChange}
                  placeholder="60"
                  style={{width: '100%', border: 'none', background: 'transparent', textAlign: 'center'}}
                />
              </td>
              <td className="label-cell gray-label">역량지수</td>
              <td className="number-cell">
                <input
                  type="number"
                  name="capacity"
                  value={formData.capacity}
                  onChange={handleInputChange}
                  placeholder="2"
                  style={{width: '100%', border: 'none', background: 'transparent', textAlign: 'center'}}
                />
              </td>
              <td className="empty-cell"></td>
              <td className="empty-cell"></td>
              <td className="empty-cell"></td>
              <td className="add-button-cell" colSpan="2">
                <button type="button" onClick={addCompetencyField} className="table-add-btn">
                  + 분야 추가
                </button>
              </td>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="label-cell">분야</td>
              <td className="label-cell">학위역량</td>
              <td className="label-cell">구성요인</td>
              <td className="label-cell">역량시수</td>
              <td className="label-cell">반영비율</td>
              <td className="label-cell">교수학습방법</td>
              <td className="label-cell">측정/평가</td>
              <td className="label-cell">평가횟수</td>
              <td className="label-cell delete-header-cell">삭제</td>
            </tr>
            {competencyFields.map((field) => (
              <tr key={field.id}>
                <td className="input-cell">
                  <input
                    type="text"
                    value={field.fieldName}
                    onChange={(e) => handleCompetencyChange(field.id, 'fieldName', e.target.value)}
                    placeholder="예) 숫자"
                    style={{width: '100%', border: 'none', background: 'transparent', textAlign: 'center'}}
                  />
                </td>
                <td className="input-cell">
                  <input
                    type="text"
                    value={field.learningCompetency}
                    onChange={(e) => handleCompetencyChange(field.id, 'learningCompetency', e.target.value)}
                    placeholder="예) 전공기초역량"
                    style={{width: '100%', border: 'none', background: 'transparent'}}
                  />
                </td>
                <td className="input-cell">
                  <textarea
                    value={field.competencyFactor}
                    onChange={(e) => handleCompetencyChange(field.id, 'competencyFactor', e.target.value)}
                    placeholder="예) 이론적 사고 능력"
                    rows={2}
                    style={{width: '100%', border: 'none', background: 'transparent', resize: 'none'}}
                  />
                </td>
                <td className="number-cell">
                  <input
                    type="number"
                    value={field.learningHours}
                    onChange={(e) => handleCompetencyChange(field.id, 'learningHours', e.target.value)}
                    placeholder="30"
                    style={{width: '100%', border: 'none', background: 'transparent', textAlign: 'center'}}
                  />
                </td>
                <td className="number-cell">
                  <input
                    type="number"
                    value={field.classificationRatio}
                    onChange={(e) => handleCompetencyChange(field.id, 'classificationRatio', e.target.value)}
                    placeholder="50"
                    style={{width: '100%', border: 'none', background: 'transparent', textAlign: 'center'}}
                  />
                </td>
                <td className="input-cell">
                  <textarea
                    value={field.teachingMethod}
                    onChange={(e) => handleCompetencyChange(field.id, 'teachingMethod', e.target.value)}
                    placeholder="예) 강의식 수업: 이론 설명"
                    rows={2}
                    style={{width: '100%', border: 'none', background: 'transparent', resize: 'none'}}
                  />
                </td>
                <td className="input-cell">
                  <textarea
                    value={field.evaluation}
                    onChange={(e) => handleCompetencyChange(field.id, 'evaluation', e.target.value)}
                    placeholder="예) 시험, 과제"
                    rows={2}
                    style={{width: '100%', border: 'none', background: 'transparent', resize: 'none'}}
                  />
                </td>
                <td className="number-cell">
                  <input
                    type="number"
                    value={field.evaluationCount}
                    onChange={(e) => handleCompetencyChange(field.id, 'evaluationCount', e.target.value)}
                    placeholder="2"
                    style={{width: '100%', border: 'none', background: 'transparent', textAlign: 'center'}}
                  />
                </td>
                <td className="delete-cell">
                  {competencyFields.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeCompetencyField(field.id)}
                      className="remove-field-btn"
                      title="분야 삭제"
                    >
                      ✕
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* 교과목 개요 */}
        <div className="section-title">교과목 개요</div>
        <div className="textarea-section">
          <div className="overview-label">교과목개요 및 특징</div>
          <textarea
            name="courseOverview"
            value={formData.courseOverview}
            onChange={handleInputChange}
            placeholder="예) 본 교과목은 학생들이 해당 분야의 기초 이론과 실무 능력을 체계적으로 학습할 수 있도록 구성되었습니다. 이론과 실습을 병행하여 학습자의 이해도를 높이고, 실제 현장에서 활용할 수 있는 실무 능력을 배양하는 것을 목표로 합니다."
            rows={4}
          />
        </div>

        {/* 수업목표 */}
        <div className="section-title">수업목표</div>
        <div className="textarea-section">
          <textarea
            name="objectives"
            value={formData.objectives}
            onChange={handleInputChange}
            placeholder="예) 1. 해당 분야의 기본 개념과 원리를 이해할 수 있다.
2. 학습한 이론을 실제 문제 해결에 적용할 수 있다.
3. 창의적 사고를 통해 새로운 아이디어를 도출할 수 있다."
            rows={3}
          />
        </div>

        {/* 주교재 */}
        <div className="section-title">주교재</div>
        <div className="textarea-section">
          <textarea
            name="mainTextbook"
            value={formData.mainTextbook}
            onChange={handleInputChange}
            rows={4}
          />
        </div>

        {/* 부교재 */}
        <div className="section-title">부교재</div>
        <div className="textarea-section">
          <textarea
            name="subTextbook"
            value={formData.subTextbook}
            onChange={handleInputChange}
            rows={4}
          />
        </div>

        {/* 참고문헌 */}
        <div className="section-title">참고문헌</div>
        <div className="textarea-section">
          <textarea
            name="references"
            value={formData.references}
            onChange={handleInputChange}
            rows={4}
          />
        </div>

        {/* 주차별 수업계획 */}
        <div className="section-title">주차별 수업계획 (1~15주차)</div>
        <table className="weekly-plan-table">
          <thead>
            <tr>
              <th className="week-col">주차</th>
              <th>수업주제 및 내용</th>
              <th>수업방법</th>
              <th>학생성장(역량제고) 전략</th>
            </tr>
          </thead>
          <tbody>
            {weeklyPlans.map((plan) => (
              <tr key={plan.week}>
                <td className="week-col">{plan.week}주차</td>
                <td>
                  <textarea
                    value={plan.topic}
                    onChange={(e) =>
                      handleWeeklyPlanChange(plan.week, 'topic', e.target.value)
                    }
                    rows={2}
                    placeholder="예) 교과목 소개 및 수업 운영 안내"
                  />
                </td>
                <td>
                  <textarea
                    value={plan.method}
                    onChange={(e) =>
                      handleWeeklyPlanChange(plan.week, 'method', e.target.value)
                    }
                    rows={2}
                    placeholder="예) 강의, 토의, 실습 등"
                  />
                </td>
                <td>
                  <textarea
                    value={plan.strategy}
                    onChange={(e) =>
                      handleWeeklyPlanChange(plan.week, 'strategy', e.target.value)
                    }
                    rows={2}
                    placeholder="예) 피드백 제공, 협동학습, 프로젝트 기반 학습 등"
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* 하단 안내 문구 */}
        <div className="footer-notice">
          "위 교재는 도서관 강의교재지원 및 추천자료실에서 열람가능"
        </div>

        {/* 제출 버튼 */}
        <div className="submit-section">
          <button type="submit" className="submit-btn">수업계획서 저장</button>
        </div>
      </form>
    </div>
  </div>

  {/* 오른쪽: 저장된 계획서 목록 */}
  <StudyPlanSave 
    savedPlans={savedPlans}
    selectedPlanId={selectedPlanId}
    onSelectPlan={handleSelectPlan}
    onDeletePlan={handleDeletePlan}
    onNewPlan={handleNewPlan}
  />
</div>
  );
}