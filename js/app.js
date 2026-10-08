// ========================================================
// DAVID YEONWOO PARK - SENIOR OPERATIONS & SYSTEMS SPECIALIST
// TRANSPILATION OF APP.JSX (STANDALONE ZERO-CORS PURE JAVASCRIPT)
// ========================================================

// ========================================================
// DAVID YEONWOO PARK - SENIOR OPERATIONS & SYSTEMS SPECIALIST
// REACT APPLICATION COMPONENTS & INTERACTIVE CONTROLLERS
// ========================================================

const {
  useState,
  useEffect,
  useRef
} = React;
// Icons
const SparklesIcon = () => /*#__PURE__*/React.createElement("svg", {
  xmlns: "http://www.w3.org/2000/svg",
  width: "18",
  height: "18",
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: "2",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  className: "inline-block"
}, /*#__PURE__*/React.createElement("path", {
  d: "M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"
}));

// ==========================================
// Visualized Diagrams (Architectural Proofs)
// ==========================================

// 1. PTP 3-Way Matching Architecture
const PtpVisualDiagram = ({
  lang
}) => {
  const labels = {
    kr: {
      title: "SAP ERP PTP 3-Way 대조 & 차액 조율 아키텍처",
      sub: "Reconciliation Workflow",
      po: "1. Purchase Order (PO)",
      gr: "2. Goods Receipt (GR)",
      inv: "3. Vendor Invoice (청구)",
      grSub: "실물 입고 확인",
      centerEngine: "SAP 3-Way 자동 대사 엔진",
      discrepancy: "⚠️ Discrepancy Flag: +RM 1,500",
      protocol: "David의 선제적 분쟁 조율 프로토콜 (Resolution)",
      step1: "• 1차 긴급 지급: PO 기준 RM 10,500 선결제 집행 (공급망 단절 방어)",
      step2: "• 2차 정산 감사: 운임/수량 오차 대조 후 4시간 내 Credit Note 공식 요청",
      keyInsight: "단순 승인 지연이 아닌, 검증된 PO 금액을 선지급하고 차액을 분리 감사하여 공급선 납품 중단 리스크를 100% 방어하는 전문 재무 정산 프로세스입니다."
    },
    en: {
      title: "SAP ERP PTP 3-Way Matching & Discrepancy Protocol",
      sub: "Reconciliation Workflow",
      po: "1. Purchase Order (PO)",
      gr: "2. Goods Receipt (GR)",
      inv: "3. Vendor Invoice",
      grSub: "Physical Warehouse Check",
      centerEngine: "SAP 3-Way Automated Audit Engine",
      discrepancy: "⚠️ Discrepancy Flag: +RM 1,500",
      protocol: "David's Proactive Dispute Resolution Protocol",
      step1: "• 1st Action: Release verified RM 10,500 PO payout (Preserving supply chain)",
      step2: "• 2nd Action: Isolate RM 1,500 freight variance, issue Credit Note request in 4h",
      keyInsight: "Rather than stalling payments, we immediately remit approved PO funds while auditing discrepancies separately, defending against supplier disruptions by 100%."
    },
    jp: {
      title: "SAP ERP PTP 3-Way対査 & 差額調整アーキテクチャ",
      sub: "Reconciliation Workflow",
      po: "1. 発注書 (PO)",
      gr: "2. 入庫受領書 (GR)",
      inv: "3. 請求書 (Vendor Invoice)",
      grSub: "実物入庫確認済",
      centerEngine: "SAP 3-Way 自動照合エンジン",
      discrepancy: "⚠️ 差額検知フラグ: +RM 1,500",
      protocol: "Davidの先行型紛争調停プロトコル (Resolution)",
      step1: "• 第1次仮払: PO基準 RM 10,500 先行支払実行（サプライチェーン遮断を防御）",
      step2: "• 第2次精査: 運賃誤差を照合の上、4時間以内にクレジットノート発行を正式要請",
      keyInsight: "承認遅延による納品停止を回避するため、確認済みのPO金額を即時仮払いし、差額のみを分離監査することで供給リスクを100%遮断する高度な財務実務です。"
    }
  };
  const l = labels[lang] || labels.kr;
  return /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between space-y-4 h-full"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between border-b border-slate-800 pb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-indigo-300 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-code-compare text-indigo-400"
  }), /*#__PURE__*/React.createElement("span", null, l.title)), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] bg-indigo-950 text-indigo-300 px-2.5 py-0.5 rounded-full border border-indigo-500/30 font-mono font-bold"
  }, l.sub)), /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-inner"
  }, /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 520 310",
    className: "w-full h-auto max-h-[290px]",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("linearGradient", {
    id: "boxGrad1",
    x1: "0",
    y1: "0",
    x2: "1",
    y2: "1"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#1e293b"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "1%",
    stopColor: "#0f172a"
  }))), /*#__PURE__*/React.createElement("rect", {
    x: "20",
    y: "20",
    width: "135",
    height: "60",
    rx: "10",
    fill: "url(#boxGrad1)",
    stroke: "#6366f1",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "87",
    y: "44",
    fill: "#818cf8",
    fontSize: "10.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.po), /*#__PURE__*/React.createElement("text", {
    x: "87",
    y: "64",
    fill: "#ffffff",
    fontSize: "12",
    fontWeight: "900",
    textAnchor: "middle"
  }, "RM 10,500 (SAP)"), /*#__PURE__*/React.createElement("rect", {
    x: "190",
    y: "20",
    width: "135",
    height: "60",
    rx: "10",
    fill: "url(#boxGrad1)",
    stroke: "#06b6d4",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "257",
    y: "44",
    fill: "#38bdf8",
    fontSize: "10.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.gr), /*#__PURE__*/React.createElement("text", {
    x: "257",
    y: "64",
    fill: "#ffffff",
    fontSize: "11",
    fontWeight: "700",
    textAnchor: "middle"
  }, l.grSub), /*#__PURE__*/React.createElement("rect", {
    x: "360",
    y: "20",
    width: "140",
    height: "60",
    rx: "10",
    fill: "url(#boxGrad1)",
    stroke: "#f43f5e",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "430",
    y: "44",
    fill: "#fb7185",
    fontSize: "10.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.inv), /*#__PURE__*/React.createElement("text", {
    x: "430",
    y: "64",
    fill: "#ffffff",
    fontSize: "12",
    fontWeight: "900",
    textAnchor: "middle"
  }, "RM 12,000"), /*#__PURE__*/React.createElement("path", {
    d: "M 87 80 L 87 115 L 210 135",
    stroke: "#6366f1",
    strokeWidth: "2",
    strokeDasharray: "3 3"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M 257 80 L 257 120",
    stroke: "#06b6d4",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M 430 80 L 430 115 L 310 135",
    stroke: "#f43f5e",
    strokeWidth: "2",
    strokeDasharray: "3 3"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "140",
    y: "120",
    width: "240",
    height: "60",
    rx: "12",
    fill: "#1e1b4b",
    stroke: "#818cf8",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "145",
    fill: "#a5b4fc",
    fontSize: "11.5",
    fontWeight: "900",
    textAnchor: "middle"
  }, l.centerEngine), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "165",
    fill: "#f87171",
    fontSize: "11",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.discrepancy), /*#__PURE__*/React.createElement("path", {
    d: "M 260 180 L 260 205",
    stroke: "#a5b4fc",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "30",
    y: "205",
    width: "460",
    height: "75",
    rx: "12",
    fill: "url(#boxGrad1)",
    stroke: "#10b981",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "228",
    fill: "#34d399",
    fontSize: "12",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.protocol), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "249",
    fill: "#e2e8f0",
    fontSize: "10.5",
    textAnchor: "middle"
  }, l.step1), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "267",
    fill: "#e2e8f0",
    fontSize: "10.5",
    textAnchor: "middle"
  }, l.step2))), /*#__PURE__*/React.createElement("div", {
    className: "bg-indigo-950/50 p-3 rounded-xl border border-indigo-500/30 text-xs text-slate-300"
  }, /*#__PURE__*/React.createElement("strong", {
    className: "text-indigo-400"
  }, "\uD83D\uDCA1 ", lang === 'jp' ? '図解のポイント:' : lang === 'en' ? 'Core Architecture Insight:' : '시각화 핵심:'), " ", l.keyInsight));
};

// 2. OTA Trouble Shooting Architecture
const AgodaVisualDiagram = ({
  lang
}) => {
  const labels = {
    kr: {
      title: "OTA 긴급 분쟁 에스컬레이션 & 환불 구조도",
      sub: "Incident Flowchart",
      s1: "Step 1. 고객 클레임 접수",
      s1Sub: "노쇼 위약금 청구 항의",
      s2: "Step 2. 시스템 로그 대조",
      s2Sub: "OTA API ↔ Hotel PMS",
      s3: "Step 3. 과실 원인 규명",
      s3Sub: "호텔 측 동기화 누락 확인",
      centerAction: "David의 긴급 승인 에스컬레이션",
      centerSub: "100% 전액 환불 즉시 승인 + 대체 숙소 보상",
      resolution: "Closed-Loop Resolution & Partner Reconcile",
      res1: "• 고객 케어: 해당 언어 공식 사죄문(謝罪文) 발송 및 CSAT 5점 만점 방어",
      res2: "• 파트너 정산: 호텔 측 위약금 면제 조항 발효 및 재발 방지 통보",
      keyInsight: "격앙된 고객에게 즉각 100% 환불을 선확약하고, 호텔 PMS 동기화 로그를 통해 불필요한 핑퐁 시간을 80% 단축시키는 글로벌 CS 전문 프로세스입니다."
    },
    en: {
      title: "OTA Emergency Dispute Escalation & Refund Matrix",
      sub: "Incident Flowchart",
      s1: "Step 1. Claim Receipt",
      s1Sub: "No-Show Penalty Protest",
      s2: "Step 2. Log Audit",
      s2Sub: "OTA API ↔ Hotel PMS",
      s3: "Step 3. Root Cause Found",
      s3Sub: "Partner Sync Failure Verified",
      centerAction: "David's Rapid Escalation Release",
      centerSub: "100% Immediate Refund + Compensation Voucher",
      resolution: "Closed-Loop Resolution & Partner Reconcile",
      res1: "• Guest Care: Formal Apology Letter & CSAT 5.0 Recovery",
      res2: "• Partner Payout: Invoking waiver clauses to clear penalty charges",
      keyInsight: "Guarantees 100% relief to the traveler while proving PMS synchronization faults through timestamped logs, reducing resolution ping-pong by 80%."
    },
    jp: {
      title: "OTA緊急紛争エスカレーション & 返金フロー構造図",
      sub: "Incident Flowchart",
      s1: "Step 1. クレーム受付",
      s1Sub: "ノーショー違約金請求への抗議",
      s2: "Step 2. システムログ照合",
      s2Sub: "OTA API ↔ Hotel PMS",
      s3: "Step 3. 過失原因の立証",
      s3Sub: "ホテル側同期漏れの特定",
      centerAction: "Davidの緊急承認エスカレーション",
      centerSub: "100%即時全額返金承認 + 代替宿泊バウチャー",
      resolution: "クローズドループ解決 & パートナー精算調停",
      res1: "• 顧客ケア: 日本語正式謝罪文の即時発行とCSAT 5.0満点防衛",
      res2: "• 提携精算: ホテル側違約金免除条項の発効および再発防止策通達",
      keyInsight: "顧客へ即座に100%返金を確約して不信感を払拭し、ホテルPMSログ照合により不毛な責任転嫁を80%短縮するグローバルOTA特化の実務です。"
    }
  };
  const l = labels[lang] || labels.kr;
  return /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between space-y-4 h-full"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between border-b border-slate-800 pb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-emerald-300 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-arrows-split-up-and-left text-emerald-400"
  }), /*#__PURE__*/React.createElement("span", null, l.title)), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] bg-emerald-950 text-emerald-300 px-2.5 py-0.5 rounded-full border border-emerald-500/30 font-mono font-bold"
  }, l.sub)), /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-inner"
  }, /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 520 310",
    className: "w-full h-auto max-h-[290px]",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("linearGradient", {
    id: "boxGrad2",
    x1: "0",
    y1: "0",
    x2: "1",
    y2: "1"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#1e293b"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#0f172a"
  }))), /*#__PURE__*/React.createElement("rect", {
    x: "20",
    y: "20",
    width: "140",
    height: "60",
    rx: "10",
    fill: "url(#boxGrad2)",
    stroke: "#f43f5e",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "90",
    y: "44",
    fill: "#fb7185",
    fontSize: "10",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.s1), /*#__PURE__*/React.createElement("text", {
    x: "90",
    y: "64",
    fill: "#ffffff",
    fontSize: "11",
    fontWeight: "600",
    textAnchor: "middle"
  }, l.s1Sub), /*#__PURE__*/React.createElement("rect", {
    x: "190",
    y: "20",
    width: "140",
    height: "60",
    rx: "10",
    fill: "url(#boxGrad2)",
    stroke: "#06b6d4",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "44",
    fill: "#38bdf8",
    fontSize: "10",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.s2), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "64",
    fill: "#ffffff",
    fontSize: "11",
    fontWeight: "600",
    textAnchor: "middle"
  }, l.s2Sub), /*#__PURE__*/React.createElement("rect", {
    x: "360",
    y: "20",
    width: "140",
    height: "60",
    rx: "10",
    fill: "url(#boxGrad2)",
    stroke: "#eab308",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "430",
    y: "44",
    fill: "#facc15",
    fontSize: "10",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.s3), /*#__PURE__*/React.createElement("text", {
    x: "430",
    y: "64",
    fill: "#ffffff",
    fontSize: "11",
    fontWeight: "600",
    textAnchor: "middle"
  }, l.s3Sub), /*#__PURE__*/React.createElement("path", {
    d: "M 160 50 L 190 50",
    stroke: "#94a3b8",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M 330 50 L 360 50",
    stroke: "#94a3b8",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M 430 80 L 430 115 L 360 135",
    stroke: "#facc15",
    strokeWidth: "2",
    strokeDasharray: "3 3"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "130",
    y: "120",
    width: "260",
    height: "60",
    rx: "12",
    fill: "#064e3b",
    stroke: "#34d399",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "145",
    fill: "#6ee7b7",
    fontSize: "11.5",
    fontWeight: "900",
    textAnchor: "middle"
  }, l.centerAction), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "165",
    fill: "#ffffff",
    fontSize: "10.5",
    fontWeight: "700",
    textAnchor: "middle"
  }, l.centerSub), /*#__PURE__*/React.createElement("path", {
    d: "M 260 180 L 260 205",
    stroke: "#34d399",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "30",
    y: "205",
    width: "460",
    height: "75",
    rx: "12",
    fill: "url(#boxGrad2)",
    stroke: "#10b981",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "228",
    fill: "#34d399",
    fontSize: "12",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.resolution), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "249",
    fill: "#e2e8f0",
    fontSize: "10.5",
    textAnchor: "middle"
  }, l.res1), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "267",
    fill: "#e2e8f0",
    fontSize: "10.5",
    textAnchor: "middle"
  }, l.res2))), /*#__PURE__*/React.createElement("div", {
    className: "bg-emerald-950/50 p-3 rounded-xl border border-emerald-500/30 text-xs text-slate-300"
  }, /*#__PURE__*/React.createElement("strong", {
    className: "text-emerald-400"
  }, "\uD83D\uDCA1 ", lang === 'jp' ? '図解のポイント:' : lang === 'en' ? 'Core Architecture Insight:' : '시각화 핵심:'), " ", l.keyInsight));
};

// 3. Operations Improvement Dashboard Architecture
const DashboardVisualDiagram = ({
  lang
}) => {
  const labels = {
    kr: {
      title: "운영 개선 대시보드 & 이슈 상태 통제 아키텍처",
      sub: "Scalable Ops Matrix",
      st1: "1. 반복 이슈 감지",
      st1Sub: "결산 지연 2~3일",
      st2: "2. 통제 체크리스트",
      st2Sub: "D-1 ~ D+3 일일 대사",
      st3: "3. 실시간 상태 지표",
      st3Sub: "SLA 80% 경보 트리거",
      st4: "4. 재사용 자산",
      st4Sub: "팀 공유 플레이북",
      monitorHead: "OPERATIONS CONTROL RADAR & ISSUE MONITOR",
      m1Name: "PTP 결산 리드타임",
      m1Val: "-35% 단축 달성",
      m2Name: "미결 이슈 누락 방어율",
      m2Val: "100% (0건 누락)",
      boxHead: "개인의 꼼꼼함을 팀의 지속 가능한 시스템 자산으로 확장",
      b1: "• 신규 온보딩 시 인수인계 시간 70% 절감 · 전 부서 표준 템플릿 배포",
      b2: "• 1회성 마감에 그치지 않고 반복 이슈의 원인을 시스템 체크리스트로 격리",
      keyInsight: "개인의 감각에 의존하던 결산 체크를 4단계 표준 프로세스로 정립하여 결산 시간을 35% 단축하고 팀 전체가 재사용 가능한 운영 자산으로 만든 실적입니다."
    },
    en: {
      title: "Operations Improvement Dashboard & Issue Control Matrix",
      sub: "Scalable Ops Matrix",
      st1: "1. Issue Detection",
      st1Sub: "2-3 Day Lag Identified",
      st2: "2. Control Checklist",
      st2Sub: "D-1 to D+3 Daily Audit",
      st3: "3. Live KPI Status",
      st3Sub: "80% SLA Alert Trigger",
      st4: "4. Reusable Asset",
      st4Sub: "Team SOP Playbook",
      monitorHead: "OPERATIONS CONTROL RADAR & ISSUE MONITOR",
      m1Name: "PTP Closing Lead Time",
      m1Val: "-35% Accelerated",
      m2Name: "Unresolved Issue Defense",
      m2Val: "100% Zero-Omission",
      boxHead: "Scaling Personal Diligence into an Organizational System Asset",
      b1: "• 70% reduction in team onboarding training duration via standardized templates",
      b2: "• Isolates root causes into procedural checklists rather than temporary fixes",
      keyInsight: "Converted ad-hoc closing rituals into a 4-step framework, accelerating monthly financial closing by 35% and packaging it as a turnkey asset for the entire department."
    },
    jp: {
      title: "運営改善ダッシュボード & 課題状態統制アーキテクチャ",
      sub: "Scalable Ops Matrix",
      st1: "1. 反復課題の検知",
      st1Sub: "決算遅延 2~3日",
      st2: "2. 統制チェックリスト",
      st2Sub: "D-1 ~ D+3 日次照合",
      st3: "3. リアルタイムKPI指標",
      st3Sub: "SLA 80% 警告発火",
      st4: "4. 再利用可能な資産",
      st4Sub: "チーム共有SOP",
      monitorHead: "OPERATIONS CONTROL RADAR & ISSUE MONITOR",
      m1Name: "PTP決算リードタイム",
      m1Val: "-35% 短縮達成",
      m2Name: "未処理課題の漏れ防止率",
      m2Val: "100% (漏れゼロ)",
      boxHead: "個人の几帳面さを、チーム全体の持続可能なシステム資産へ昇華",
      b1: "• 新規参画者の引き継ぎ工数を70%削減・標準テンプレートを全社共有",
      b2: "• 単発の帳尻合わせではなく、反復課題の根本原因をチェックリストで恒久排除",
      keyInsight: "属人的な勘に頼っていた決算確認を4段階の標準運用に昇華させ、決算時間を35%短縮したチーム共有のオペレーション資産です。"
    }
  };
  const l = labels[lang] || labels.kr;
  return /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between space-y-4 h-full"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between border-b border-slate-800 pb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-300 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-chart-line text-cyan-400"
  }), /*#__PURE__*/React.createElement("span", null, l.title)), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] bg-cyan-950 text-cyan-300 px-2.5 py-0.5 rounded-full border border-cyan-500/30 font-mono font-bold"
  }, l.sub)), /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-inner"
  }, /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 520 310",
    className: "w-full h-auto max-h-[290px]",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("linearGradient", {
    id: "boxGrad3",
    x1: "0",
    y1: "0",
    x2: "1",
    y2: "1"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#1e293b"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "1%",
    stopColor: "#0f172a"
  }))), /*#__PURE__*/React.createElement("rect", {
    x: "15",
    y: "15",
    width: "110",
    height: "50",
    rx: "8",
    fill: "url(#boxGrad3)",
    stroke: "#f43f5e",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "70",
    y: "34",
    fill: "#fb7185",
    fontSize: "9.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.st1), /*#__PURE__*/React.createElement("text", {
    x: "70",
    y: "52",
    fill: "#ffffff",
    fontSize: "10.5",
    fontWeight: "700",
    textAnchor: "middle"
  }, l.st1Sub), /*#__PURE__*/React.createElement("path", {
    d: "M 125 40 L 145 40",
    stroke: "#94a3b8",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "145",
    y: "15",
    width: "115",
    height: "50",
    rx: "8",
    fill: "url(#boxGrad3)",
    stroke: "#6366f1",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "202",
    y: "34",
    fill: "#818cf8",
    fontSize: "9.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.st2), /*#__PURE__*/React.createElement("text", {
    x: "202",
    y: "52",
    fill: "#ffffff",
    fontSize: "10.5",
    fontWeight: "700",
    textAnchor: "middle"
  }, l.st2Sub), /*#__PURE__*/React.createElement("path", {
    d: "M 260 40 L 280 40",
    stroke: "#94a3b8",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "280",
    y: "15",
    width: "110",
    height: "50",
    rx: "8",
    fill: "url(#boxGrad3)",
    stroke: "#06b6d4",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "335",
    y: "34",
    fill: "#38bdf8",
    fontSize: "9.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.st3), /*#__PURE__*/React.createElement("text", {
    x: "335",
    y: "52",
    fill: "#ffffff",
    fontSize: "10.5",
    fontWeight: "700",
    textAnchor: "middle"
  }, l.st3Sub), /*#__PURE__*/React.createElement("path", {
    d: "M 390 40 L 410 40",
    stroke: "#94a3b8",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "410",
    y: "15",
    width: "95",
    height: "50",
    rx: "8",
    fill: "url(#boxGrad3)",
    stroke: "#10b981",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "457",
    y: "34",
    fill: "#34d399",
    fontSize: "9.5",
    fontWeight: "800",
    textAnchor: "middle"
  }, l.st4), /*#__PURE__*/React.createElement("text", {
    x: "457",
    y: "52",
    fill: "#ffffff",
    fontSize: "10.5",
    fontWeight: "700",
    textAnchor: "middle"
  }, l.st4Sub), /*#__PURE__*/React.createElement("rect", {
    x: "25",
    y: "80",
    width: "470",
    height: "85",
    rx: "12",
    fill: "#081b30",
    stroke: "#00f0ff",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "45",
    y: "102",
    fill: "#5ee7d0",
    fontSize: "10",
    fontWeight: "900"
  }, l.monitorHead), /*#__PURE__*/React.createElement("rect", {
    x: "390",
    y: "90",
    width: "85",
    height: "18",
    rx: "4",
    fill: "#10b981"
  }), /*#__PURE__*/React.createElement("text", {
    x: "432",
    y: "103",
    fill: "#042f2e",
    fontSize: "9.5",
    fontWeight: "900",
    textAnchor: "middle"
  }, "HEALTH: 99.4%"), /*#__PURE__*/React.createElement("text", {
    x: "45",
    y: "127",
    fill: "#94a3b8",
    fontSize: "10"
  }, l.m1Name), /*#__PURE__*/React.createElement("rect", {
    x: "160",
    y: "120",
    width: "160",
    height: "8",
    rx: "4",
    fill: "#1e293b"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "160",
    y: "120",
    width: "135",
    height: "8",
    rx: "4",
    fill: "#6366f1"
  }), /*#__PURE__*/React.createElement("text", {
    x: "330",
    y: "127",
    fill: "#818cf8",
    fontSize: "10",
    fontWeight: "700"
  }, l.m1Val), /*#__PURE__*/React.createElement("text", {
    x: "45",
    y: "148",
    fill: "#94a3b8",
    fontSize: "10"
  }, l.m2Name), /*#__PURE__*/React.createElement("rect", {
    x: "160",
    y: "141",
    width: "160",
    height: "8",
    rx: "4",
    fill: "#1e293b"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "160",
    y: "141",
    width: "160",
    height: "8",
    rx: "4",
    fill: "#10b981"
  }), /*#__PURE__*/React.createElement("text", {
    x: "330",
    y: "148",
    fill: "#34d399",
    fontSize: "10",
    fontWeight: "700"
  }, l.m2Val), /*#__PURE__*/React.createElement("rect", {
    x: "25",
    y: "180",
    width: "470",
    height: "75",
    rx: "12",
    fill: "url(#boxGrad3)",
    stroke: "#38bdf8",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "204",
    fill: "#38bdf8",
    fontSize: "11",
    fontWeight: "900",
    textAnchor: "middle"
  }, l.boxHead), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "225",
    fill: "#e2e8f0",
    fontSize: "10",
    textAnchor: "middle"
  }, l.b1), /*#__PURE__*/React.createElement("text", {
    x: "260",
    y: "243",
    fill: "#cbd5e1",
    fontSize: "10",
    textAnchor: "middle"
  }, l.b2))), /*#__PURE__*/React.createElement("div", {
    className: "bg-cyan-950/50 p-3 rounded-xl border border-cyan-500/30 text-xs text-slate-300"
  }, /*#__PURE__*/React.createElement("strong", {
    className: "text-cyan-400"
  }, "\uD83D\uDCA1 ", lang === 'jp' ? '図解のポイント:' : lang === 'en' ? 'Core Architecture Insight:' : '시각화 핵심:'), " ", l.keyInsight));
};

// ==========================================
// Main Application Component
// ==========================================
function App() {
  const [lang, setLang] = useState('kr');
  const [selectedProject, setSelectedProject] = useState('ptp');
  const [contactModalOpen, setContactModalOpen] = useState(false);
  const [filterCategory, setFilterCategory] = useState('all');
  const [notification, setNotification] = useState(null);

  // BPO Simulator State
  const [simDomain, setSimDomain] = useState('ptp');
  const [userAnswer, setUserAnswer] = useState('');
  const [simResult, setSimResult] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const showToast = msg => {
    setNotification(msg);
    setTimeout(() => setNotification(null), 3000);
  };

  // ==========================================
  // Chatbot Simulator State & Multilingual Logic (KR / EN / JP)
  // ==========================================
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [activeCategory, setActiveCategory] = useState('auto');
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [attachedFile, setAttachedFile] = useState(null);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Multilingual Dictionary for Chatbot UI & Responses

  const [messages, setMessages] = useState([{
    sender: 'ai',
    text: botI18n[lang]?.welcome || botI18n.kr.welcome
  }]);

  // Sync initial welcome greeting when site language changes
  useEffect(() => {
    setMessages(prev => {
      if (prev.length === 1 && prev[0].sender === 'ai') {
        return [{
          sender: 'ai',
          text: botI18n[lang]?.welcome || botI18n.kr.welcome
        }];
      }
      return prev;
    });
  }, [lang]);
  useEffect(() => {
    if (isChatOpen) {
      chatEndRef.current?.scrollIntoView({
        behavior: 'smooth'
      });
    }
  }, [messages, isTyping, isChatOpen]);
  const handleFileSelect = e => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      alert(lang === 'jp' ? 'ファイルサイズは10MB以下のみ添付可能です。' : lang === 'en' ? 'File size must be 10MB or less.' : '파일 크기는 10MB 이하만 첨부 가능합니다.');
      return;
    }
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = ev => {
        setAttachedFile({
          name: file.name,
          size: file.size,
          type: file.type,
          previewUrl: ev.target.result
        });
      };
      reader.readAsDataURL(file);
    } else {
      setAttachedFile({
        name: file.name,
        size: file.size,
        type: file.type || 'application/pdf',
        previewUrl: null
      });
    }
  };
  const getAiResponse = (userText, selectedCat, attachment, activeLang) => {
    const lower = userText.toLowerCase().replace(/\s+/g, ' ').trim();

    // Detect response language: prioritize Japanese/English in text or site lang
    let targetLang = activeLang || 'kr';
    if (/[\u3040-\u30ff]/.test(userText)) {
      targetLang = 'jp';
    } else if (/^[a-zA-Z0-9\s.,!?\'\"#\-()\/]+$/.test(userText.trim()) && userText.trim().length > 3) {
      targetLang = 'en';
    }

    // 1. Attached File Processing (Screenshot, Image, PDF)
    if (attachment) {
      const fileName = attachment.name.toLowerCase();
      const isPdf = attachment.type && attachment.type.includes('pdf');
      const sizeKb = (attachment.size / 1024).toFixed(1);

      // 1-A. Flight / Hotel waiver proof (결항, 병원, 진단서, 호텔, 항공, 취소)
      if (fileName.includes('결항') || fileName.includes('진단서') || fileName.includes('병원') || fileName.includes('소견') || fileName.includes('欠航') || fileName.includes('診断') || fileName.includes('cancellation') || fileName.includes('medical') || fileName.includes('hotel') || fileName.includes('flight') || selectedCat === 'hotel' || selectedCat === 'flight' || lower.includes('호텔') || lower.includes('항공') || lower.includes('특가') || lower.includes('취소') || lower.includes('ホテル') || lower.includes('航空') || lower.includes('flight')) {
        if (targetLang === 'jp') {
          return "📎 **[添付 " + (isPdf ? 'PDF文書' : 'スクショ証憑') + " 受領：特別免除（Waiver）審査開始]**\n\n" + "ご提出いただいたファイル `" + attachment.name + "` (" + sizeKb + " KB) を安全に電算登録いたしました。\n\n" + "• **証憑分析結果**:\n" + "  - 発行日、公的機関印、不可抗力事由（欠航通知書/医師診断書/現地PMS連動エラー画面）の照合完了。\n" + "  - 約款上「返金不可」の予約であっても、本公的証憑に基づき**提携先（ホテル/航空会社）へのペナルティ免除（Waiver）稟議**を即時上程いたします。\n\n" + "• **今後のスケジュール**:\n" + "  - 提携先総支配人および本部審査チームへ免除要請公文を発信完了。\n" + "  - 審査確定後、**全額返金またはVIP補償クレジット付与**を執行し、1時間以内にメールにて最終結果をご案内いたします。";
        } else if (targetLang === 'en') {
          return "📎 **[Attached " + (isPdf ? 'PDF Document' : 'Screenshot Proof') + " Received: Waiver Escalation Initiated]**\n\n" + "Your document `" + attachment.name + "` (" + sizeKb + " KB) has been securely logged.\n\n" + "• **Evidence Audit Result**:\n" + "  - Verified official stamps, issuance dates, and objective force majeure grounds (flight cancellation notice / medical diagnosis / system sync failure).\n" + "  - Even under 'Non-Refundable' conditions, this official proof qualifies for an immediate **partner penalty waiver escalation**.\n\n" + "• **Next Action Steps**:\n" + "  - Formal waiver petition submitted directly to hotel GM & airline headquarters.\n" + "  - Final resolution (**100% full refund or VIP goodwill travel credits**) will be confirmed via email within 1 hour.";
        } else {
          return "📎 **[첨부 " + (isPdf ? 'PDF 서류' : '스크린샷/이미지 증빙') + " 접수 및 예외 구제(Waiver) 심사 착수]**\n\n" + "제출해 주신 파일 `" + attachment.name + "` (" + sizeKb + " KB)이 보안 전산망에 정상 등록되었습니다.\n\n" + "• **증빙 분석 결과**:\n" + "  - 첨부 서류의 발행 일자, 기관 직인, 객관적 불가항력 사유(결항 통보서/의사 소견서/현장 전산 오류 캡처) 대조 완료.\n" + "  - 약관상 '환불 불가' 조건이더라도 본 공적 증빙을 근거로 **파트너사(호텔/항공사) 페널티 면제(Waiver) 에스컬레이션**이 즉시 승인 심사에 상정됩니다.\n\n" + "• **후속 조치 일정**:\n" + "  - 제휴처 총지배인 및 본사 심사팀에 면제 공문 발송 완료.\n" + "  - 심사 결과에 따라 **전액 환불 또는 VIP 보상 크레딧 전환**이 최종 집행되며, 1시간 이내 알림톡/이메일로 확정 회신드립니다.";
        }
      }

      // 1-B. BPO PTP / Freight variance / Invoice proof
      if (fileName.includes('인보이스') || fileName.includes('invoice') || fileName.includes('운임') || fileName.includes('sap') || fileName.includes('정산') || fileName.includes('請求') || fileName.includes('運賃') || selectedCat === 'payout' || lower.includes('정산') || lower.includes('운임') || lower.includes('精算') || lower.includes('運賃')) {
        if (targetLang === 'jp') {
          return "📊 **[財務照合証憑（請求書・運賃B/L）ファイル受領確認]**\n\n" + "アップロードされたファイル `" + attachment.name + "` (" + sizeKb + " KB) をSAP ERP対査システムへ登録いたしました。\n\n" + "• **3-Way対査照合結果**:\n" + "  - 請求インボイスとWMS検収伝票の運賃差額区間の電算照合を完了。\n" + "  - 取引先の資金繰り防衛のため、**承認済みPO確定額は本日15:00までに先行決済を執行**いたします。\n" + "  - 不整合差額は本証憑に基づき**クレジットノート発行および財務相殺処理**を確定いたしました。";
        } else if (targetLang === 'en') {
          return "📊 **[Financial Audit Document (Invoice / Freight B/L) Received]**\n\n" + "Uploaded file `" + attachment.name + "` (" + sizeKb + " KB) is registered in SAP ERP reconciliation module.\n\n" + "• **3-Way Matching Audit Result**:\n" + "  - Verified freight variance between invoiced charges and WMS goods receipt logs.\n" + "  - To prevent supply chain disruption, **verified PO approved funds are scheduled for upfront payout today by 3:00 PM**.\n" + "  - Discrepancy is isolated for formal **Credit Note issuance and ledger reconciliation**.";
        } else {
          return "📊 **[정산 대조 서류(인보이스/운임 B/L) 파일 접수 확인]**\n\n" + "업로드하신 파일 `" + attachment.name + "` (" + sizeKb + " KB)이 SAP ERP 전산 대조 모듈에 등록되었습니다.\n\n" + "• **3-Way 대조 분석 결과**:\n" + "  - 제출하신 청구 인보이스와 WMS 입고 검수 전표의 운임 단가 차액 구간을 전산 매칭하였습니다.\n" + "  - 공급업체 유동성 보호를 위해 **정상 승인된 PO 확정 대금은 금일 15:00까지 선결제 집행**됩니다.\n" + "  - 불일치 차액은 본 증빙을 근거로 **Credit Note(감액 전표) 발행 및 회계 상계 처리**가 확정되었습니다.";
        }
      }

      // 1-C. Chargeback / Signed POD proof
      if (fileName.includes('pod') || fileName.includes('서명') || fileName.includes('차지백') || fileName.includes('受領') || selectedCat === 'dispute' || lower.includes('차지백') || lower.includes('도용') || lower.includes('チャージバック')) {
        if (targetLang === 'jp') {
          return "🛡️ **[不正決済異議申立 - 配達受領書（POD）証憑受領確認]**\n\n" + "ご提出いただいた配達証憑ファイル `" + attachment.name + "` をカード紛争防衛システムへ登録いたしました。\n\n" + "• **反証照合結果**:\n" + "  - 3Dセキュア2.0認証ログと受領サインが一致し、国際カードブランドのイシュアー責任転換（Liability Shift）要件を完全充足。\n" + "  - **24時間以内にVisa/Mastercardへ正式異議申立（Representment）を発信**し、加盟店の保留売上金を即時解除いたします。";
        } else if (targetLang === 'en') {
          return "🛡️ **[Chargeback Defense - Proof of Delivery (POD) Verified]**\n\n" + "Signed POD document `" + attachment.name + "` is logged into merchant dispute defense module.\n\n" + "• **Dispute Representment Verification**:\n" + "  - 3D Secure 2.0 auth token matches signed courier delivery slip, fulfilling card brand Liability Shift requirements.\n" + "  - **Formal Representment dispatched to Visa/Mastercard within 24h**, and merchant frozen reserves are released immediately.";
        } else {
          return "🛡️ **[부정결제 소명 - 배송완료 서명(POD) 증빙 접수 확인]**\n\n" + "제출하신 배송 증빙 파일 `" + attachment.name + "`이 카드사 분쟁 방어 시스템에 등록되었습니다.\n\n" + "• **소명 대조 결과**:\n" + "  - 3D Secure 2.0 인증 로그와 수령인 자필 서명이 일치하여 카드사 발급사 책임 전환(Liability Shift) 요건을 완벽히 충족합니다.\n" + "  - **24시간 이내 Visa/Mastercard 공식 이의제기(Representment)**를 발송하며, 가맹점 정산 보류금을 즉시 해제 조치합니다.";
        }
      }

      // 1-D. General Attachment response
      if (targetLang === 'jp') {
        return "📎 **[証憑書類の受領完了 - " + attachment.name + "]**\n\n" + "お送りいただいた証憑ファイル（`" + attachment.name + "`, " + sizeKb + " KB）を正常に受信いたしました。\n\n" + "• **実務審査プロセス**:\n" + "  - 朴（David）の実務運営基準に基づき、添付ファイルの規格および正当性を照合しております。\n" + "  - 証憑の有効性が確認された場合、通常の手数料を免除した**特例救済（Waiver）または差額調整プロトコル**を執行いたします。\n\n" + "追加の注文番号・予約番号がございましたら併せてご入力ください。";
      } else if (targetLang === 'en') {
        return "📎 **[Evidence Intake Confirmed - " + attachment.name + "]**\n\n" + "Your document (`" + attachment.name + "`, " + sizeKb + " KB) has been safely received.\n\n" + "• **Audit Procedure**:\n" + "  - Auditing document authenticity against platform SLA terms and liability rules.\n" + "  - Upon validation, **penalty waiver or ledger reconciliation protocols** will be authorized immediately.\n\n" + "Please provide any related booking or invoice numbers if available.";
      } else {
        return "📎 **[증빙 자료 접수 완료 - " + attachment.name + "]**\n\n" + "보내주신 증빙 (`" + attachment.name + "`, " + sizeKb + " KB)이 정상 수신되었습니다.\n\n" + "• **실무 심사 절차**:\n" + "  - David의 실무 운영 프로세스에 따라 첨부 파일의 규격 및 진위 여부를 대조하고 있습니다.\n" + "  - 증빙이 유효할 경우 일반 취소 수수료 부과 없이 **예외 승인(Waiver) 또는 차액 조정 프로토콜**이 가동됩니다.\n\n" + "추가로 문의하실 주문/예약 번호가 있으시면 함께 입력해 주시기 바랍니다.";
      }
    }

    // 2. Policy Challenge / Blind Cancellation Verification
    // e.g. "정책 안 보고 바로 취소되나요?", "Can I cancel without checking policy?", "ポリシー未確認で即時キャンセルされるのですか？"
    const hasPolicyTerm = lower.includes('정책') || lower.includes('규정') || lower.includes('약관') || lower.includes('기준') || lower.includes('ポリシー') || lower.includes('規程') || lower.includes('約款') || lower.includes('規約') || lower.includes('polic') || lower.includes('rule') || lower.includes('term');
    const hasBlindTerm = lower.includes('안 보') || lower.includes('안보') || lower.includes('무조건') || lower.includes('그냥') || lower.includes('바로') || lower.includes('확인 없') || lower.includes('확인안') || lower.includes('見ず') || lower.includes('確認せず') || lower.includes('無条件') || lower.includes('without') || lower.includes('blind') || lower.includes('immediately');
    const isBlindCancel = hasPolicyTerm && hasBlindTerm || lower.includes('바로 취소') && (lower.includes('정책') || lower.includes('안') || lower.includes('규정') || lower.includes('되나요') || lower.includes('되는 건가요')) || lower.includes('바로 취소가 되는 건가요') || lower.includes('即時キャンセルされるのですか');
    if (isBlindCancel) {
      if (targetLang === 'jp') {
        return "⚠️ **[運営リスク管理およびポリシー遵守原則のご案内]**\n\n" + "**いいえ、即時取消は行いません。** すべての注文・予約の取消および返金は、**購入された品目カテゴリー（宿泊/航空/配送/BPO財務）と各約款規定（取消期限、返金不可特約等）を事前に厳格照合（SOP検証）した上でのみ承認可否を決定**いたします。\n\n" + "規約確認を怠った無条件の即時承認は、提携先・パートナー契約違反および会社に対する甚大な財務損失（不当返金の累積）を招くため、断じて執行されません。\n\n" + "正確な精査のため、以下の項目をご確認ください：\n" + "• **品目・サービス分類**: 宿泊（ホテル）/ 航空券 / 実物配送 / 仕入先精算\n" + "• **予約・配送ステータス**: 返金不可特約、物流倉庫発送完了の有無、取消可能期限の超過状況\n" + "• **証憑書類の添付**: 下部 📎 ボタンより欠航証明書、医師診断書、請求書等を添付いただければ、特約免除（Waiver）審査が可能です。\n\n" + "上部カテゴリーを選択いただくか、詳細状況をお知らせいただければ、実務意思決定ツリーに基づき正確な取消規定をご案内いたします。";
      } else if (targetLang === 'en') {
        return "⚠️ **[Operational Risk Management & Policy Governance Notice]**\n\n" + "**No, absolutely not.** All cancellations and refunds are approved **only after strict pre-audit against product categories (Hotel/Flight/Retail/BPO) and respective SLA terms (cancellation deadlines, non-refundable clauses)**.\n\n" + "Blind, unverified immediate approvals violate partner merchant agreements and expose the company to catastrophic financial liabilities (accumulative fraudulent refunds).\n\n" + "Please provide the following details for accurate assessment:\n" + "• **Product Category**: Accommodation (Hotel) / Flights / Physical Goods / Vendor Payout\n" + "• **Order/Booking Status**: Non-refundable promotional rate, WMS dispatch stage, cancellation cutoff deadline\n" + "• **Document Evidence**: Use the 📎 button below to attach flight cancellation slips, medical certificates, or invoices for waiver review.\n\n" + "Select a category tab above or state your specific product details to trigger David's operational decision tree.";
      } else {
        return "⚠️ **[운영 리스크 관리 및 정책 검토 원칙 안내]**\n\n" + "아닙니다. 모든 주문 및 예약의 취소/환불은 **구매하신 상품의 유형(숙박/항공/배송/BPO)과 개별 약관(취소 가능 시한, 환불불가 조건 등)을 사전에 엄격히 대조(SOP 검증)**한 후에만 승인 여부가 결정됩니다.\n\n" + "정책 검토 없는 무조건적인 즉시 승인은 제휴사/파트너 계약 위반 및 회사에 막대한 재무 손실(부당 환불 누적)을 초래하므로 절대로 집행되지 않습니다.\n\n" + "정확한 분석을 위해 아래 항목을 확인해 주시기 바랍니다:\n" + "• **상품/서비스 유형**: 숙박(호텔) / 항공권 / 실물 배송 / 벤더 정산\n" + "• **주문/예약 상태**: 환불불가 특가 여부, 물류센터 출고 완료 여부, 취소 가능 시한 경과 여부\n" + "• **증빙 서류 첨부**: 하단 📎 버튼을 통해 결항증명서, 의사진단서, 인보이스 등을 첨부하시면 예외 면제(Waiver) 심사가 가능합니다.\n\n" + "상단 카테고리 탭을 선택하시거나 상품 유형과 세부 상태를 말씀해 주시면, David의 실무 운영 의사결정 체계(Decision Tree)에 따라 정확한 환불/위약금 규정을 분석해 드립니다.";
      }
    }

    // 3. Determine Domain / Product Category
    let domain = selectedCat;
    if (domain === 'auto') {
      if (lower.includes('호텔') || lower.includes('숙박') || lower.includes('숙소') || lower.includes('객실') || lower.includes('체크인') || lower.includes('hotel') || lower.includes('ota') || lower.includes('투숙') || lower.includes('오버부킹') || lower.includes('pms') || lower.includes('宿泊')) {
        domain = 'hotel';
      } else if (lower.includes('항공') || lower.includes('비행기') || lower.includes('탑승') || lower.includes('노쇼') || lower.includes('flight') || lower.includes('fare') || lower.includes('공항세') || lower.includes('발권') || lower.includes('航空') || lower.includes('搭乗')) {
        domain = 'flight';
      } else if (lower.includes('배송') || lower.includes('출고') || lower.includes('택배') || lower.includes('송장') || lower.includes('반품') || lower.includes('물류') || lower.includes('shipping') || lower.includes('delivery') || lower.includes('커머스') || lower.includes('주문') || lower.includes('配送') || lower.includes('出荷') || lower.includes('返品')) {
        domain = 'commerce';
      } else if (lower.includes('정산') || lower.includes('지연') || lower.includes('운임') || lower.includes('인보이스') || lower.includes('sap') || lower.includes('3-way') || /\\bpo\\b/.test(lower) || lower.includes('발주') || lower.includes('크레딧') || lower.includes('차액') || lower.includes('variance') || lower.includes('대금') || lower.includes('精算') || lower.includes('遅延') || lower.includes('運賃') || lower.includes('請求書') || lower.includes('payout')) {
        domain = 'payout';
      } else if (lower.includes('차지백') || lower.includes('도용') || lower.includes('부정') || lower.includes('분쟁') || lower.includes('이의제기') || lower.includes('chargeback') || lower.includes('3ds') || lower.includes('pod') || lower.includes('チャージバック') || lower.includes('不正利用')) {
        domain = 'dispute';
      } else if (lower.includes('환불 불가') || lower.includes('환불불가') || lower.includes('특가') || lower.includes('non-refundable') || lower.includes('返金不可')) {
        domain = 'hotel';
      }
    }

    // Category 1: Accommodation / Hotel (OTA)
    if (domain === 'hotel') {
      if (lower.includes('환불 불가') || lower.includes('환불불가') || lower.includes('특가') || lower.includes('non-refundable') || lower.includes('프로모션') || lower.includes('오버부킹') || lower.includes('시스템 오류') || lower.includes('返金不可') || lower.includes('オーバーブッキング')) {
        if (targetLang === 'jp') {
          return "🏨 **[宿泊・ホテル - 返金不可（Non-refundable）特約例外審査SOP]**\n\n" + "• **基本原則**: プロモーション特約付き返金不可プランは、成立時点で100%の取消手数料が適用され、自己都合キャンセルは原則不可となります。\n\n" + "• **例外救済（CX Recovery）プロトコル**:\n" + "  1) **プラットフォーム・施設側帰責時**: ホテルPMS連携エラーやオーバーブッキング等でお客様過失0%が確定した場合、手数料100%免除の全額返金＋近隣5つ星代替宿泊無償提供＋VIP特別補償バウチャー付与。\n" + "  2) **不可抗力事由**: 航空便欠航や医師診断書の提出により、総支配人宛のWaiver（違約金免除）特例を審査。\n\n" + "関連書類がございましたら、**下部 📎 ボタン**より添付いただければ即座に本部エスカレーションを実施いたします。";
        } else if (targetLang === 'en') {
          return "🏨 **[Hotel / Accommodation - Non-Refundable Rate Exception SOP]**\n\n" + "• **Standard Policy**: Non-refundable promotional bookings incur 100% cancellation penalties upon confirmation; voluntary cancellation is restricted.\n\n" + "• **Waiver & CX Recovery Protocol**:\n" + "  1) **Platform / Hotel Liability Verified**: In cases of PMS API sync outages or property overbooking (0% guest liability), we execute a 100% full refund + complimentary 5-star re-accommodation + VIP goodwill travel voucher.\n" + "  2) **Force Majeure / Medical Emergency**: Validated flight cancellations or medical certifications are escalated for GM waiver authorization.\n\n" + "Please attach supporting evidence via the **📎 button below** for immediate headquarters escalation.";
        } else {
          return "🏨 **[숙박/호텔 - 환불 불가(Non-refundable) 특가 예외 심사 SOP]**\n\n" + "• **원칙 규정**: 특가 프로모션으로 체결된 환불 불가 상품은 예약 즉시 100% 취소 수수료가 적용되어 단순 변심 취소가 불가합니다.\n\n" + "• **예외 승인(CX Recovery) 프로토콜**:\n" + "  1) **플랫폼/호텔 귀책 입증 시**: 호텔 PMS-플랫폼 간 API 연동 오류나 오버부킹 등으로 고객 과실 0%가 확인된 경우, 수수료 100% 면제 전액 환불 + 인근 5성급 대체 숙소 전액 지원 + VIP 보상 바우처 지급.\n" + "  2) **불가항력 사유**: 천재지변(항공 결항, 자연재해) 또는 직계가족 질병 공식 증빙 접수 시 호텔 총지배인 Waiver(위약금 면제) 공문 심사 후 승인.\n\n" + "관련 증빙(PDF/스크린샷)이 있으시면 **하단 📎 버튼**으로 첨부해 주시면 즉시 본사 에스컬레이션을 가동하겠습니다.";
        }
      }
      if (targetLang === 'jp') {
        return "🏨 **[宿泊・ホテル - 標準取消および返金規定案内]**\n\n" + "• **無料取消期間内（チェックイン24〜48時間前まで）**: 取消手数料0円にて決済代金100%全額返金となります。\n" + "• **取消期限超過・当日取消**: 宿泊約款に基づき初日1泊分のペナルティが控除された後、残額が返金されます。\n" + "• **ノーショー（無断不泊）**: 原則全額返金不可となります。\n\n" + "チェックイン予定日および施設名をお知らせいただければ、正確な手数料発生区分を算定いたします。";
      } else if (targetLang === 'en') {
        return "🏨 **[Hotel / Accommodation - Standard Cancellation Guidelines]**\n\n" + "• **Within Free Cancellation Window (24-48h prior to check-in)**: 100% full refund with zero penalty fees.\n" + "• **After Cutoff / Same-Day Cancellation**: The first night's room rate penalty is deducted per hotel policy, and remaining balance is refunded.\n" + "• **No-Show**: 100% non-refundable under standard lodging regulations.\n\n" + "Please provide your check-in date and property name to determine exact penalty thresholds.";
      } else {
        return "🏨 **[숙박/호텔 - 취소 및 환불 규정 심사 안내]**\n\n" + "• **무료 취소 기한 내 (체크인 24~48시간 전)**: 위약금 없이 결제 금액 100% 전액 환불 처리됩니다.\n" + "• **취소 마감 시한 경과 / 당일 취소**: 호텔 숙박 약관상 첫 1박 요금 상당의 취소 수수료(페널티)가 공제된 후 잔액이 환불됩니다.\n" + "• **노쇼(No-Show)**: 원칙상 전액 환불 불가 처리됩니다.\n\n" + "체크인 예정일자와 호텔명을 알려주시면 정확한 취소 수수료 부과 구간을 산출해 드립니다.";
      }
    }

    // Category 2: Flight / Aviation
    if (domain === 'flight') {
      if (targetLang === 'jp') {
        return "✈️ **[航空券 - 航空会社運賃規則（Fare Rules）に基づく払戻案内]**\n\n" + "• **航空会社運賃約款の優先適用**: 航空券の払戻は、航空会社の運賃クラス別規程および発券代理店手数料に基づき算定されます。\n" + "• **控除項目**: [航空会社取消手数料] + [発券手数料]を控除した残額が返金対象となります。\n" + "• **全額返金対象**: 燃油サーチャージおよび未使用の空港諸税（Tax）は運賃種別に関わらず100%返金されます。\n" + "• **当日・ノーショー（無断不搭乗）の注意**: 出発後の取消はノーショー違約金が加算されるため、出発前の事前申請が必須です。\n\n" + "欠航や急病などの不可抗力事由は、**下部 📎 ボタン**より証明書を添付いただければ免除（Waiver）審査が可能です。";
      } else if (targetLang === 'en') {
        return "✈️ **[Flights - Airline Fare Rules & Ticketing Refund Guidelines]**\n\n" + "• **Governed by Airline Fare Rules**: Flight refunds strictly adhere to the carrier's fare class tariff and ticketing agency service terms.\n" + "• **Penalty Deduction**: [Airline Cancellation Fee] + [Ticketing Agency Fee] will be deducted from the refund payout.\n" + "• **Guaranteed 100% Refund Items**: Fuel surcharges and unused government airport taxes are refunded 100% regardless of fare class.\n" + "• **Same-Day / No-Show Notice**: Cancellations after departure incur severe no-show fees; advance notice before takeoff is mandatory.\n\n" + "In case of flight delays or medical issues, attach proof via the **📎 button below** for airline waiver review.";
      } else {
        return "✈️ **[항공권 - 항공사 운임 규정(Fare Rules) 기반 환불 안내]**\n\n" + "• **항공사 운임 규정 우선 적용**: 항공권 환불은 항공사 자체 운임 등급(Class) 및 발권 여행사 수수료 약관이 복합 적용됩니다.\n" + "• **수수료 공제 구조**: [항공사 취소 위약금] + [발권 대행 수수료]가 공제된 후 잔여 금액이 환불됩니다.\n" + "• **100% 환불 보장 항목**: 유류할증료 및 미사용 공항시설이용료(Tax)는 규정에 관계없이 전액 환불됩니다.\n" + "• **출발 당일/No-Show 주의**: 출발 시각 이후 취소 시 노쇼 페널티가 추가 부과되므로 반드시 출발 전 사전 취소 접수가 필수입니다.\n\n" + "결항이나 질병 등 불가항력 취소 시 **하단 📎 버튼**으로 증빙(결항확인서 등)을 첨부하시면 위약금 면제(Waiver) 심사가 가능합니다.";
      }
    }

    // Category 3: Commerce / Physical Goods Delivery
    if (domain === 'commerce') {
      if (lower.includes('출고 전') || lower.includes('준비 중') || lower.includes('배송 전') || lower.includes('出荷前') || lower.includes('pre-dispatch') || lower.includes('before shipping')) {
        if (targetLang === 'jp') {
          return "📦 **[配送・EC - 出荷前即時キャンセル承認]**\n\n" + "WMS（物流倉庫）ピッキングおよび配送業者引渡前の「出荷準備中」段階であれば、**往復運賃等の控除なく即時100%全額返金**を執行いたします。\n\n" + "注文番号をお知らせいただければ、物流システム上で即時出荷ロックを適用いたします。";
        } else if (targetLang === 'en') {
          return "📦 **[E-Commerce / Logistics - Pre-Dispatch Instant Cancellation]**\n\n" + "For orders confirmed in 'Preparing for Shipment' prior to WMS packing and courier handover, **immediate 100% cancellation is processed with zero return freight deductions**.\n\n" + "Please provide your order number to apply an immediate shipping lock in our warehouse system.";
        } else {
          return "📦 **[물류/커머스 - 출고 전 즉시 취소 승인]**\n\n" + "WMS(물류창고) 패킹 및 택배사 인계 전 '배송 준비 중' 상태로 확인되는 경우, **별도 배송비 차감 없이 즉시 100% 전액 결제 취소**가 집행됩니다.\n\n" + "주문번호를 입력해 주시면 실시간 WMS 출고 상태를 조회하여 락(Lock)을 걸어드립니다.";
        }
      }
      if (targetLang === 'jp') {
        return "📦 **[配送・EC - 出荷完了商品の返品・返金規定]**\n\n" + "• **出荷完了（配送中・配達完了）**: すでに運送会社へ引き渡された荷物は即時取消が不可となり、受領後の**「返品検品手続き」**となります。\n" + "• **お客様都合の返品**: 未開封状態の倉庫検品後、**往復運賃（配送費）を控除**した残額を返金いたします。\n" + "• **誤配送・破損・不良**: 弊社全額負担にて無償集荷および全額返金・交換対応を実施。\n\n" + "破損や誤配送の場合は、**下部 📎 ボタン**より写真または送り状を添付いただければ、即座に無償返品を受理いたします。";
      } else if (targetLang === 'en') {
        return "📦 **[E-Commerce / Logistics - Dispatched Goods Return & Refund Terms]**\n\n" + "• **Post-Dispatch (In Transit / Delivered)**: Packages handed over to couriers cannot be canceled immediately and must follow standard **return inspection protocols**.\n" + "• **Change of Mind**: **Round-trip shipping fees are deducted** from the refund balance upon unopened inspection at warehouse.\n" + "• **Damaged / Defective / Wrong Item**: 100% covered by platform with complimentary reverse pickup and full refund.\n\n" + "For damage or defect claims, attach product photos via the **📎 button below** for immediate courier claim processing.";
      } else {
        return "📦 **[물류/커머스 - 출고 완료 상품 반품 및 환불 규정]**\n\n" + "• **출고 완료(배송 중 / 배송 완료)**: 물류센터에서 이미 택배사로 인계된 상품은 즉시 취소가 불가하며, 상품 수령 후 **'반품 검수 절차'**로 진행됩니다.\n" + "• **단순 변심 반품**: 포장 미개봉 검수 완료 후 **왕복 배송비(운임)를 차감**한 잔액이 환불됩니다.\n" + "• **오배송 / 파손 하자**: 당사 100% 부담으로 무상 회수 및 전액 환불/교환 집행.\n\n" + "파손이나 오배송의 경우 **하단 📎 버튼**으로 제품 사진/송장 스샷을 첨부해 주시면 즉각 무상 반품이 접수됩니다.";
      }
    }

    // Category 4: BPO PTP Accounts Payable & Freight Variance
    if (domain === 'payout' || lower.includes('정산') || lower.includes('지연') || lower.includes('운임') || lower.includes('인보이스') || lower.includes('sap') || lower.includes('3-way') || /\\bpo\\b/.test(lower) || lower.includes('발주') || lower.includes('차액') || lower.includes('精算') || lower.includes('運賃') || lower.includes('payout')) {
      if (targetLang === 'jp') {
        return "📊 **[BPO PTP - サプライチェーン防衛・運賃差額精算プロトコル]**\n\n" + "• **供給停止リスクの完全遮断**: SAP ERP 3-Way対査において運賃差額（Variance）が検出された場合でも、仕入先の納品停止を防ぐため**「確認済みのPO承認金額」を本日15:00までに先行決済（Pre-payment）執行**いたします。\n\n" + "• **差額分離監査と相殺処理**: 不整合差額は別勘定で分離監査を実施し、公式証憑に基づく**クレジットノート発行要請および翌月相殺処理**にて決算期限を遵守します。\n\n" + "請求書やB/Lファイルを**下部 📎 ボタン**より添付いただければ、ERP対査および先行決済連動を迅速に実施いたします。";
      } else if (targetLang === 'en') {
        return "📊 **[BPO PTP - Supply Chain Safeguard & Freight Variance Protocol]**\n\n" + "• **Averting Delivery Freeze**: When a freight variance is flagged during SAP ERP 3-Way matching, we execute an **upfront payout of verified PO funds today by 3:00 PM** to prevent supplier shutdown.\n\n" + "• **Variance Isolation & Credit Note**: Discrepancies are isolated for independent financial audit, requesting a formal **Credit Note for ERP ledger reconciliation**.\n\n" + "Attach invoices or B/L slips via the **📎 button below** to expedite automated ERP ledger matching.";
      } else {
        return "📊 **[BPO PTP - 공급망 방어 및 운임 차액 정산 프로토콜]**\n\n" + "• **공급망 중단 리스크 차단**: SAP ERP 3-Way 대조 중 운임 차액(Variance)이 발견되더라도, 협력사 납품 중단을 방지하기 위해 **'확인된 정상 PO 승인 금액'은 금일 15:00까지 우선 선결제(Pre-payment) 집행**합니다.\n\n" + "• **차액 분리 감사 및 상계**: 불일치 운임 차액은 별도 계정으로 분리 감사하며, 공식 증빙 기반 **Credit Note(감액 전표) 발행 요청 및 익월 상계 처리**로 재무 결산 마감을 준수합니다.\n\n" + "인보이스 또는 B/L 파일을 **하단 📎 버튼**으로 첨부해 주시면 ERP 전산 원장 대조 및 긴급 선결제 승인이 즉시 연동됩니다.";
      }
    }

    // Category 5: Chargeback Dispute & Fraud Defense
    if (domain === 'dispute' || lower.includes('차지백') || lower.includes('도용') || lower.includes('부정') || lower.includes('분쟁') || lower.includes('3ds') || lower.includes('チャージバック')) {
      if (targetLang === 'jp') {
        return "🛡️ **[加盟店保護 - 3DS 2.0 チャージバック反証プロトコル]**\n\n" + "• **イシュアー責任転換（Liability Shift）適用**: 3Dセキュア2.0認証済み取引は国際カードブランド（Visa/Mastercard）規程に基づき発券銀行へ責任が転換されるため、加盟店責務は免責となります。\n\n" + "• **24時間以内の異議申立（Representment）**: 配達受領署名（POD）および認証ログを基に迅速反証を行い、加盟店の**保留売上金を即時解除**いたします。\n\n" + "受領書（POD）や領収書を**下部 📎 ボタン**よりアップロードいただければ、24時間以内に反証手続きを完遂いたします。";
      } else if (targetLang === 'en') {
        return "🛡️ **[Merchant Protection - 3DS 2.0 Chargeback Defense Protocol]**\n\n" + "• **Issuer Liability Shift Applied**: Verified 3D Secure 2.0 transactions transfer fraud liability to the issuing bank under Visa/Mastercard mandates, exempting merchant fault.\n\n" + "• **24-Hour Dispute Representment**: Armed with signed Proof of Delivery (POD) and device auth logs, we submit formal counter-claims within 24h to **unfreeze merchant payout reserves**.\n\n" + "Attach signed PODs or transaction receipts via the **📎 button below** for dispute submission.";
      } else {
        return "🛡️ **[가맹점 리스크 방어 - 3DS 2.0 차지백 소명 프로토콜]**\n\n" + "• **책임 전환(Liability Shift) 적용**: 3D Secure 2.0 본인 인증 거래는 Visa/Mastercard 국제 규정에 따라 카드 발급사(Issuer)로 책임이 전환되므로 가맹점 책임이 면책됩니다.\n\n" + "• **24시간 내 Representment(이의 제기)**: 배송 완료 서명 증빙(POD)과 인증 로그를 바탕으로 신속 소명하여 가맹점의 **정산 보류금(Reserve)을 즉시 해제**합니다.\n\n" + "배송완료 서명(POD) 또는 주문 영수증을 **하단 📎 버튼**으로 업로드해 주시면 24시간 내 소명 서류가 발송됩니다.";
      }
    }

    // Category 6: General refund / cancellation mentioned without category
    if (lower.includes('환불') || lower.includes('취소') || lower.includes('refund') || lower.includes('cancel') || lower.includes('返金') || lower.includes('キャンセル')) {
      if (targetLang === 'jp') {
        return "💡 **[品目別取消・返金のご案内]**\n\n" + "お問い合わせの返金規定は、**対象の品目分類により約款が異なります**:\n\n" + "1. 🏨 **宿泊・ホテル**: 取消期限（24〜48時間前）照合 / 返金不可特約の例外救済\n" + "2. ✈️ **航空券**: 航空会社運賃約款（Fare Rules）および発券手数料控除\n" + "3. 📦 **配送・EC**: 出荷前全額取消 vs 発送後往復運賃控除返品\n" + "4. 📊 **BPO財務・精算**: SAP 3-Way運賃差額分離および先行決済執行\n" + "5. 🛡️ **決済紛争**: 3Dセキュア2.0責任転換の反証手続\n\n" + "関連証憑（領収書、診断書、欠航証明、請求書）がございましたら**下部 📎 ボタン**より添付いただければ迅速に審査いたします。";
      } else if (targetLang === 'en') {
        return "💡 **[Cancellation & Refund Guidelines by Category]**\n\n" + "Refund protocols are **governed by specific category terms**:\n\n" + "1. 🏨 **Hotel / Lodging**: Cutoff deadlines (24-48h prior) / Non-refundable rate waiver review\n" + "2. ✈️ **Flights**: Airline Fare Rules tariffs & agency fee deductions\n" + "3. 📦 **E-Commerce**: Pre-dispatch 100% cancel vs in-transit return shipping fee\n" + "4. 📊 **BPO Finance**: SAP 3-Way variance isolation & upfront payout approval\n" + "5. 🛡️ **Payment Dispute**: 3DS 2.0 Liability Shift representment\n\n" + "Select a category above or attach proof via the **📎 button below** for faster processing.";
      } else {
        return "💡 **[상품 유형별 취소 및 환불 안내]**\n\n" + "문의하신 환불 규정은 **어떤 상품군인지에 따라 적용 약관이 다릅니다**:\n\n" + "1. 🏨 **숙박/호텔**: 무료 취소 기한(체크인 24~48h 전) 확인 / 환불불가 특가 상품 예외 심사\n" + "2. ✈️ **항공권**: 항공사 운임 규정(Fare Rules) 및 발권 수수료 차감\n" + "3. 📦 **실물 배송**: 물류센터 출고 전(100% 즉시 취소) vs 출고 후(왕복 배송비 차감 반품)\n" + "4. 📊 **BPO 정산**: SAP 3-Way 대조 운임 차액 분리 및 확인액 선결제 집행\n" + "5. 🛡️ **결제 분쟁**: 3D Secure 2.0 발급사 책임전환 소명\n\n" + "관련 증빙(영수증, 진단서, 결항증명, 인보이스)이 있으시면 **하단 📎 버튼**으로 첨부해 주시면 더욱 신속한 심사가 가능합니다.";
      }
    }

    // General fallback / greeting
    return botI18n[targetLang]?.welcome || botI18n.kr.welcome;
  };
  const handleSendMessage = (customText, customAttachment) => {
    const fileToSend = customAttachment !== undefined ? customAttachment : attachedFile;
    let textToSend = typeof customText === 'string' ? customText : inputValue;
    if (!fileToSend && (!textToSend || !textToSend.trim()) || isTyping) return;
    if (!textToSend || !textToSend.trim()) {
      textToSend = lang === 'jp' ? "📎 証憑書類 [" + fileToSend.name + "] 添付照会" : lang === 'en' ? "📎 Attaching evidence document [" + fileToSend.name + "]" : "📎 증빙 파일 [" + fileToSend.name + "] 첨부 문의";
    }
    const userText = textToSend.trim();
    const newMsg = {
      sender: 'user',
      text: userText,
      attachment: fileToSend ? {
        ...fileToSend
      } : null
    };
    setMessages(prev => [...prev, newMsg]);
    setInputValue('');
    setAttachedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setIsTyping(true);
    setTimeout(() => {
      const reply = getAiResponse(userText, activeCategory, fileToSend, lang);
      setMessages(prev => [...prev, {
        sender: 'ai',
        text: reply
      }]);
      setIsTyping(false);
    }, 1500);
  };
  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Global Translations Dictionary

  const curT = t[lang] || t.kr;

  // 3 Representative Projects Data fully translated per language

  const loadSimSample = () => {
    const sampleText = bpoDomains[simDomain].sampleAnswer[lang] || bpoDomains[simDomain].sampleAnswer.kr;
    setUserAnswer(sampleText);
    showToast(lang === 'jp' ? '模範回答がエディタに自動ロードされました！' : lang === 'en' ? 'Model answer auto-loaded!' : '모범 답변이 텍스트 영역에 자동 로드되었습니다!');
  };

  // Upgraded Context-Aware, Differentiated AI Rubric & Dynamic Feedback Engine
  const evaluateAnswer = () => {
    if (!userAnswer.trim()) {
      showToast(lang === 'jp' ? '回答文を入力してください！' : lang === 'en' ? 'Please enter a response!' : '대응하실 답변 내용을 입력해 주세요!');
      return;
    }
    setIsEvaluating(true);
    setSimResult(null);
    setTimeout(() => {
      const domain = bpoDomains[simDomain];
      const lowerAnswer = userAnswer.toLowerCase().trim();
      const trimmedAnswer = userAnswer.trim();
      const len = trimmedAnswer.length;
      const cleanSnippet = len > 22 ? trimmedAnswer.slice(0, 20) + '...' : trimmedAnswer;

      // 1. Operational Pillars Detection (0 - 44 pts)
      let matchedPillars = [];
      let missingPillars = [];
      domain.pillars.forEach(p => {
        const hit = p.terms.some(t => lowerAnswer.includes(t.toLowerCase()));
        if (hit) {
          matchedPillars.push(p);
        } else {
          missingPillars.push(p);
        }
      });
      const pillarScore = matchedPillars.length * 10 + (matchedPillars.length === 4 ? 4 : 0);
      const complianceRate = Math.round(matchedPillars.length / 4 * 100);

      // 2. Language-Specific Keyword Detection for Tags
      const langKws = domain.keywords && (domain.keywords[lang] || domain.keywords.kr) || [];
      let matched = [];
      let missing = [];
      langKws.forEach(kw => {
        if (lowerAnswer.includes(kw.toLowerCase())) {
          matched.push(kw);
        } else {
          if (missing.length < 5) missing.push(kw);
        }
      });

      // 3. Detect User's Input Topic / Intent Focus
      let topic = 'general';
      if (["수수료", "비용", "fee", "charge", "cost", "commission", "手数料", "費用", "料金"].some(k => lowerAnswer.includes(k))) {
        topic = 'fee';
      } else if (["환불", "취소", "반품", "refund", "cancel", "return", "返金", "キャンセル", "返品"].some(k => lowerAnswer.includes(k))) {
        topic = 'refund';
      } else if (["일정", "언제", "기한", "시간", "날짜", "timeline", "schedule", "when", "eta", "deadline", "日程", "いつ", "期日", "時間"].some(k => lowerAnswer.includes(k))) {
        topic = 'timeline';
      } else if (["배송", "출고", "도착", "택배", "송장", "delivery", "shipment", "dispatch", "tracking", "配送", "出荷", "到着", "追跡", "伝票"].some(k => lowerAnswer.includes(k))) {
        topic = 'delivery';
      } else if (["시스템", "오류", "버그", "접속", "로그인", "장애", "system", "error", "bug", "login", "outage", "システム", "エラー", "不具合", "障害", "ログイン"].some(k => lowerAnswer.includes(k))) {
        topic = 'system';
      } else if (["정산", "대금", "지급", "입금", "송금", "결제", "payment", "settlement", "payout", "invoice", "精算", "代金", "支払", "送金", "決済"].some(k => lowerAnswer.includes(k))) {
        topic = 'settlement';
      } else if (["죄송", "사과", "미안", "apologize", "sorry", "excuse", "申し訳", "お詫び", "ご迷惑"].some(k => lowerAnswer.includes(k))) {
        topic = 'apology';
      } else if (["문의", "여부", "인가요", "있나요", "어떻게", "질문", "?", "inquiry", "how", "what", "question", "照会", "確認", "でしょうか", "質問"].some(k => lowerAnswer.includes(k))) {
        topic = 'inquiry';
      } else if (len < 18) {
        topic = 'short';
      }

      // 4. Dynamic Differentiated Score Calculation
      // A. Length baseline (10 to 22)
      let lenScore = Math.min(22, 10 + Math.round(len * 0.45));

      // B. Deterministic character-hash jitter (1 to 7)
      let charSum = 0;
      for (let i = 0; i < trimmedAnswer.length; i++) {
        charSum = (charSum + trimmedAnswer.charCodeAt(i) * (i + 1)) % 1000;
      }
      const jitter = charSum % 7 + 1;

      // C. Topic relevance bonus
      const topicBonus = topic !== 'general' && topic !== 'short' ? 5 : 0;

      // D. Professional Tone, Empathy & Courtesy (0 - 20 pts)
      let toneScoreRaw = 0;
      const greetingMarkers = ["dear", "hello", "hi", "mr", "ms", "안녕하십니까", "안녕하세요", "님", "이사님", "대표님", "담당자님", "様", "平素", "お世話"];
      if (greetingMarkers.some(m => lowerAnswer.includes(m))) toneScoreRaw += 5;
      const empathyMarkers = ["apologize", "sorry", "inconvenience", "regret", "pardon", "사과", "죄송", "불편", "양해", "유감", "우려", "お詫び", "申し訳", "ご不便", "ご迷惑", "ご心配", "陳謝"];
      if (empathyMarkers.some(m => lowerAnswer.includes(m))) toneScoreRaw += 6;
      const closingMarkers = ["sincerely", "regards", "respectfully", "thanks", "thank you", "드림", "배상", "올림", "감사합니다", "최선", "何卒", "宜しく", "敬具", "拝啓", "最善", "謹んで"];
      if (closingMarkers.some(m => lowerAnswer.includes(m))) toneScoreRaw += 5;
      if (userAnswer.includes('\n')) toneScoreRaw += 4;
      toneScoreRaw = Math.min(20, toneScoreRaw);

      // E. Numerical & Timeline Precision (0 - 18 pts)
      let precScoreRaw = 0;
      const numberMatch = userAnswer.match(/\d+/g);
      if (numberMatch && numberMatch.length >= 1) precScoreRaw += 5;
      if (numberMatch && numberMatch.length >= 2) precScoreRaw += 5;
      const actionMarkers = ["credit", "refund", "release", "restore", "unblock", "payout", "audit", "representment", "상계", "환불", "복구", "해제", "지급", "집행", "소명", "감사", "이의제기", "返金", "復旧", "解除", "送金", "相殺", "異議", "決済", "執行", "監査"];
      if (actionMarkers.some(m => lowerAnswer.includes(m))) precScoreRaw += 8;
      precScoreRaw = Math.min(18, precScoreRaw);

      // Total Composite Ops Score
      let totalScore = lenScore + jitter + topicBonus + pillarScore + toneScoreRaw + precScoreRaw;
      totalScore = Math.max(16, Math.min(99, totalScore));

      // Sub-metric percentages
      const toneScore = Math.min(100, Math.round(Math.max(20, toneScoreRaw / 20 * 100)));
      const riskScore = Math.min(100, Math.round(Math.max(25, (precScoreRaw + lenScore) / 40 * 100)));

      // 5. Context-Aware Diagnosis Badge
      let grade = "";
      let diagnosisBadge = "";
      if (totalScore >= 90) {
        grade = lang === 'jp' ? 'Senior Specialist (Sランク)' : lang === 'en' ? 'Senior Specialist (Grade S)' : 'Senior Specialist (S등급)';
        diagnosisBadge = lang === 'jp' ? 'Sランク · シニア統括水準' : lang === 'en' ? 'Grade S · Senior Executive Fit' : 'S등급 · 시니어 총괄 수준';
      } else if (totalScore >= 75) {
        grade = lang === 'jp' ? 'Professional (Aランク)' : lang === 'en' ? 'Professional (Grade A)' : 'Professional (A등급)';
        diagnosisBadge = lang === 'jp' ? 'Aランク · 実務自律遂行水準' : lang === 'en' ? 'Grade A · Autonomous Professional' : 'A등급 · 자율 실무 수행 수준';
      } else if (totalScore >= 50) {
        grade = lang === 'jp' ? 'Standard (Bランク)' : lang === 'en' ? 'Standard (Grade B)' : 'Standard (B등급)';
        diagnosisBadge = lang === 'jp' ? 'Bランク · 標準実務水準 (要補完)' : lang === 'en' ? 'Grade B · Standard (Gaps Found)' : 'B등급 · 기본 접수 수준 (보완 필요)';
      } else {
        grade = lang === 'jp' ? 'Developing (Cランク)' : lang === 'en' ? 'Developing (Grade C)' : 'Developing (C등급)';
        if (topic === 'fee') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 手数料照会受付 (SOP補完要)' : lang === 'en' ? 'Grade C · Fee Inquiry (Needs SOP)' : 'C등급 · 수수료 질의 접수 (SOP 소명 필요)';
        } else if (topic === 'refund') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 返金要請受付 (過失ゼロ明言要)' : lang === 'en' ? 'Grade C · Refund Request (Needs 0% Liability)' : 'C등급 · 환불 요청 접수 (무과실 소명 필요)';
        } else if (topic === 'timeline') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 日程照会段階 (SLA確約要)' : lang === 'en' ? 'Grade C · Timeline Inquiry (Needs SLA)' : 'C등급 · 일정 확인 단계 (SLA 확약 필요)';
        } else if (topic === 'delivery') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 配送確認段階 (POD受領証要)' : lang === 'en' ? 'Grade C · Delivery Inquiry (Needs POD)' : 'C등급 · 배송 확인 단계 (POD 증빙 필요)';
        } else if (topic === 'system') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · システム障害受付 (代替復旧要)' : lang === 'en' ? 'Grade C · System Incident (Needs Fallback)' : 'C등급 · 시스템 장애 접수 (우회 복구 필요)';
        } else if (topic === 'settlement') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 精算保留受付 (先行決済要)' : lang === 'en' ? 'Grade C · Settlement Hold (Needs Advance Payout)' : 'C등급 · 정산 보류 접수 (선결제 방어 필요)';
        } else if (topic === 'apology') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 謝罪のみ受付 (是正措置要)' : lang === 'en' ? 'Grade C · Apology Only (Needs Action)' : 'C등급 · 단순 사과 접수 (실행 조치 결여)';
        } else if (topic === 'inquiry') {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 受動的照会調 (主導性要)' : lang === 'en' ? 'Grade C · Passive Tone (Needs Ownership)' : 'C등급 · 수동 질의 어조 (오너십 필요)';
        } else {
          diagnosisBadge = lang === 'jp' ? 'Cランク · 単文入力 (完成書簡推奨)' : lang === 'en' ? 'Grade C · Brief Input (Draft Full Resolution)' : 'C등급 · 단문 입력 (완성 서신 권장)';
        }
      }

      // 6. Dynamic Contextual Summary Tailored to User's Specific Input
      let summary = "";
      if (totalScore >= 90) {
        if (lang === 'jp') {
          summary = `極めて卓越した実務運用統制力です。4大核心要件を網羅し、客観的証拠とSLA期限を両立させて組織の金銭的・関係的リスクを完璧に防衛しています。`;
        } else if (lang === 'en') {
          summary = `Exemplary operational leadership. Seamlessly integrates all key policy pillars, backing commitments with concrete SLA timelines and financial mitigation to defend business continuity.`;
        } else {
          summary = `탁월한 운영 통제력과 완벽한 리스크 방어 역량을 입증했습니다. 4대 핵심 원칙을 충실히 반영하여, 구체적 SLA 타임라인과 재무적 보호 조치를 통해 이해관계자 신뢰를 완벽히 확보했습니다.`;
        }
      } else if (totalScore >= 75) {
        if (lang === 'jp') {
          summary = `実務オペレーターとして安定した対応力です。主要な方針は押さえられており、細部の定量的SLAコミットや証拠提出プロトコルをさらに補強すればシニア統括水準へ昇格します。`;
        } else if (lang === 'en') {
          summary = `Solid professional competence. Demonstrates sound policy comprehension, with room to reach senior excellence by reinforcing quantitative SLA turnaround times.`;
        } else {
          summary = `현업 실무자로서 우수한 대응력을 보여주었습니다. 핵심 처리 방향성은 정확히 수립되었으며, 수치 기반의 명확한 SLA 데드라인과 공식 증빙 절차를 한층 더 보강하면 시니어급 완결성을 확보할 수 있습니다.`;
        }
      } else {
        // Tailored specifically to what user wrote
        if (topic === 'fee') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：重大なクレーム現場において、単なる手数料の有無照会のみでは取引先の納品停止や離脱を防ぐことはできません。システムログに基づく発生原因の客観的説明と、規定に則った手数料免除または次回インボイス相殺（Credit Note）の処理基準を先行提示する実務文が必要です。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): In high-priority escalations, a simple fee inquiry does not prevent operational standstills. You must clarify technical audit findings and formally commit to fee waiver guidelines or upcoming Credit Note ledger reconciliations in a complete executive draft.`;
          } else {
            summary = `입력하신 내용('${cleanSnippet}') 분석: 에스컬레이션 분쟁 현장에서 단순 수수료 발생 여부 확인만으로는 벤더의 납품 정지나 가맹점의 유동성 위기를 방어할 수 없습니다. 발생 원인에 대한 시스템 로그 대조 사실을 밝히고, 규정에 따른 비용 감면이나 공식 인보이스 상계(Credit Note) 처리 기준을 선제적으로 공표하는 완결형 비즈니스 서신을 작성해야 합니다.`;
          }
        } else if (topic === 'refund') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：単なる返金受付にとどまらず、システム障害による「顧客過失ゼロの明言」、手数料全額免除での100%即時返金、そして顧客離脱を防ぐ補償バウチャーを兼ね備えた実務文が必要です。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): Beyond simple refund acknowledgments, senior CX leadership requires validating 0% customer liability, 100% fee waivers, and goodwill accommodation credits to safeguard retention.`;
          } else {
            summary = `입력하신 내용('${cleanSnippet}') 분석: 단순 환불 접수 안내에 그치지 않고, 시스템 장애에 따른 '고객 과실 0% 규명'과 위약금 전액 면제(100% 환불), 그리고 고객 이탈을 막는 보상 바우처를 결합한 완결형 실무 솔루션이 제시되어야 합니다.`;
          }
        } else if (topic === 'timeline') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：「なるべく早く」といった曖昧な返答は取引先の不信を深めます。「本日15:00まで」や「15分以内」のように数値化されたSLAタイムラインの明記が不可欠です。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): Vague promises such as 'as soon as possible' escalate partner anxiety. Commit to rigid quantitative SLA benchmarks (e.g. 'by 3:00 PM today' or 'within 15 minutes').`;
          } else {
            summary = `입력하신 내용('${cleanSnippet}') 분석: 비즈니스 클레임 현장에서 모호한 일정 안내는 상대방의 불안을 가중시킵니다. '금일 15:00까지' 또는 '접수 후 15분 이내'와 같이 분·시 단위 확약 SLA 타임라인을 제시해야 비즈니스 신뢰를 구축할 수 있습니다.`;
          }
        } else if (topic === 'delivery') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：物流トラブルにおいては主観的説明ではなく、配送業者の署名受領証（POD）や送り状追跡番号などの客観的物証を援用することが決定的な防衛となります。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): Logistical disputes cannot rely on subjective explanations; cite signed Proof of Delivery (POD) and carrier tracking manifests to overturn bank claims.`;
          } else {
            summary = `입력하신 내용('${cleanSnippet}') 분석: 물류·운송 관련 분쟁에서는 주관적 설명보다 배송업체 서명 날인 수령증(POD) 및 운송장 추적 번호와 같은 객관적 물증을 인용해야 가맹점 정산 유동성을 보호할 수 있습니다.`;
          }
        } else if (topic === 'system') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：全社業務停止を伴うシステム障害では、単なる不具合の認知を超え、緊急代替トークンの即時プロビジョニングや監査ログ照合による根本原因特定が求められます。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): When critical systems fail, acknowledging the incident is insufficient; demonstrate technical governance by deploying emergency fallback tokens and audit log reviews.`;
          } else {
            summary = `입력하신 내용('${cleanSnippet}') 분석: 전산 장애로 인한 고객사 마비 상황에서는 단순 장애 시인이 아닌, 비상 우회 인증 토큰 즉각 발급이나 전산 감사 로그 조회를 통한 원인 규명 등 구체적 엔지니어링 통제력을 보여주어야 합니다.`;
          }
        } else if (topic === 'settlement') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：資金繰りに直結するクレームにおいては、全額保留を回避し、検証完了分の先行決済や正常売上の保留金解除を即時執行することが不可欠です。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): Cash flow disputes demand operational agility; release verified payout portions or unfreeze unaffected reserves immediately to avert merchant churn.`;
          } else {
            summary = `입력하신 내용('${cleanSnippet}') 분석: 자금 유동성에 직결된 클레임에서는 전액 지급 지연 대신 검증 완료된 금액의 선지급(선결제)이나 정상 매출금 보류 해제 조치를 취해야 파트너사 계약 파기를 막을 수 있습니다.`;
          }
        } else if (topic === 'apology') {
          if (lang === 'jp') {
            summary = `ご入力いただいた「${cleanSnippet}」に関する分析：迅速な謝罪と丁寧な共感姿勢は素晴らしい初動ですが、謝罪のみでは現場リスクを防衛できません。客観的原因の究明と、具体的な是正措置（先行送金・代替トークン・異議申立等）の即時提示が不可欠です。`;
          } else if (lang === 'en') {
            summary = `Analysis of your input ('${cleanSnippet}'): While courteous de-escalation is commendable, empathy alone does not resolve critical operational risk. You must ground the response with objective root cause validation and decisive financial/operational remedies.`;
          } else {
            summary = `입력하신 응답('${cleanSnippet}') 분석: 신속한 사과와 초기 공감 태도는 훌륭하나, 사과만으로는 현장 리스크를 방어할 수 없습니다. 시스템 객관적 원인 소명과 구체적인 해결 실행 조치(선결제, 토큰 재발급, 이의제기 등)가 반드시 결합되어야 합니다.`;
          }
        } else if (topic === 'inquiry') {
          if (lang === 'jp') {
            summary = `対応文のトーン診断（照会・質問調の検知:「${cleanSnippet}」）：エスカレーション対応において、シニア担当者は受動的な質問者ではなく、状況を即座に掌握して解決策を主導する「オーナシップ（当事者意識）」を発揮すべきです。`;
          } else if (lang === 'en') {
            summary = `Draft Tone Diagnosis ('${cleanSnippet}'): In executive escalations, senior operators must not ask passive questions; you must demonstrate decisive ownership with concrete operational solutions.`;
          } else {
            summary = `입력하신 대응문 어조 진단 ('${cleanSnippet}'): 비즈니스 에스컬레이션 현장에서 시니어 담당자는 수동적인 질문이나 확인 요청을 하는 사람이 아니라, 상황을 즉시 장악하고 해결책을 제시하는 '오너십(Ownership)'을 발휘해야 합니다.`;
          }
        } else {
          if (lang === 'jp') {
            summary = `ご入力いただいた検索語句「${cleanSnippet}」の分析：短いキーワードが入力されています。本シミュレーターでは、顧客や取引先に送信する完成された実務文（挨拶・原因釈明・解決措置・SLA期限・結び）を作成した際に正確な評価が行われます。`;
          } else if (lang === 'en') {
            summary = `Analysis for short input '${cleanSnippet}': A brief query was entered. For accurate competency assessment, draft full professional correspondence (Greeting - Root Cause - Resolution Actions - SLA Deadline - Sign-off).`;
          } else {
            summary = `입력하신 단문 검색어('${cleanSnippet}') 분석: 현재 단문 키워드가 입력되었습니다. 실무 시뮬레이터는 실제 고객 또는 벤더에게 발신하는 완성된 전문 비즈니스 대응문(인사말-원인소명-해결조치-SLA기한-맺음말)을 작성할 때 정확한 실무 적합성 진단이 이루어집니다.`;
          }
        }
      }

      // 7. Focused, Targeted Strengths and Gaps (Non-redundant, Max 2-3 items)
      const strengths = matchedPillars.map(p => p.strength[lang] || p.strength.kr);
      if (toneScoreRaw >= 11) {
        strengths.unshift(lang === 'jp' ? "【ビジネス礼儀】丁寧な敬語と迅速な共感姿勢により、顧客・取引先との初期信頼関係を良好に維持しています。" : lang === 'en' ? "[Business Etiquette] Professional de-escalation tone and clear empathy reassuring the client." : "[비즈니스 에티켓] 신속하고 정중한 공감대 형성을 통해 고객/벤더의 초기 불만을 효과적으로 완화했습니다.");
      }

      // Smart targeted gaps selection (Max 2~3 high-signal items instead of all 6)
      let gaps = [];
      if (totalScore < 90) {
        // A. Topic-Specific Target Gap
        if (topic === 'fee') {
          gaps.push(lang === 'jp' ? "▲ [手数料の法的根拠欠落] 手数料の有無の受動的確認ではなく、規定に基づく手数料免除または次回インボイスでの相殺（Credit Note）処理を明記してください。" : lang === 'en' ? "▲ [Fee Justification Omitted] Instead of asking about fees, explicitly cite policy-based fee waivers or upcoming invoice Credit Note reconciliations." : "▲ [수수료/비용 소명 부재] '수수료 여부' 질의에 그치지 않고, 시스템 대조 중 발생한 오차에 대한 공식 감면 또는 감액 전표(Credit Note) 발행 지침을 명문화해야 합니다.");
        } else if (topic === 'refund') {
          gaps.push(lang === 'jp' ? "▲ [過失ゼロ・手数料免除の明言] 返金手配に加え、システム連携不具合に伴う「お客様過失ゼロ」と取消手数料の完全免除を明記してください。" : lang === 'en' ? "▲ [0% Liability & Full Waiver] Explicitly affirm zero customer fault alongside full penalty fee waivers and instant refund execution." : "▲ [무과실 및 100% 환불 명시] 환불 절차와 함께 시스템 동기화 오류로 인한 고객 과실 0% 및 취소 수수료 전액 면제를 즉각 보증해야 합니다.");
        } else if (topic === 'timeline') {
          gaps.push(lang === 'jp' ? "▲ [確定目標時刻の明示] 曖昧な日程説明を排し、具体的な時刻・分単位の締め切り（例：本日15:00、24時間以内）を提示してください。" : lang === 'en' ? "▲ [Definitive Turnaround Target] Replace general timeline phrases with exact hours or SLA targets (e.g. 3:00 PM today, within 24h)." : "▲ [정량적 확약 시간 부재] 모호한 일정 표현 대신 구체적인 시 단위/분 단위 마감 시간(예: 금일 15:00, 24시간 내)을 명문화하세요.");
        } else if (topic === 'delivery') {
          gaps.push(lang === 'jp' ? "▲ [署名受領証（POD）の証拠確保] 単なる配送照会にとどまらず、カード会社への反証に必須となる署名受領証（POD）および追跡記録を明記してください。" : lang === 'en' ? "▲ [Signed POD Evidentiary Defense] Cite signed Proof of Delivery (POD) and tracking manifests rather than informal shipping remarks." : "▲ [배송 완료 서명 증빙 확보] 단순 배송 조회 안내에 그치지 않고, 카드사 반증에 필수적인 서명 날인 배송증빙(POD) 및 송장 추적 기록을 명시하세요.");
        } else if (topic === 'system') {
          gaps.push(lang === 'jp' ? "▲ [技術的緊急是正措置の欠落] システム障害の通知にとどまらず、15分以内の全社員復旧を約束する緊急SAML代替トークンプロビジョニングを明記してください。" : lang === 'en' ? "▲ [Emergency Technical Workaround] Go beyond outage notices by provisioning emergency fallback SAML tokens to restore user access within 15 minutes." : "▲ [기술적 긴급 복구책 부재] 시스템 장애 공지를 넘어, 15분 내 전원 복구를 보증하는 비상 SAML 대체 토큰 프로비저닝 조치를 선언해야 합니다.");
        } else if (topic === 'settlement') {
          gaps.push(lang === 'jp' ? "▲ [資金繰りの先行防衛] 代金支払の全面凍結を避け、照合完了済みの正常発注額（RM 10,500）の優先執行を宣言してください。" : lang === 'en' ? "▲ [Proactive Cashflow Protection] Avoid freezing full payouts; execute emergency release of verified PO amounts immediately." : "▲ [정산 유동성 선제 방어] 대금 지급을 전면 중단하지 말고, 대조 확인된 정상 발주액(RM 10,500) 우선 긴급 집행 조치를 취하세요.");
        } else if (topic === 'apology') {
          gaps.push(lang === 'jp' ? "▲ [実質的解決措置の不足] 丁寧な謝罪を超えて、取引先を実質的に安堵させる即時救済措置（先行送金、全額返金、緊急トークン発行）を提示してください。" : lang === 'en' ? "▲ [Concrete Action Remedy Required] Beyond polite apologies, commit to immediate tangible remedies (advance remittance, 100% refund, or emergency token release)." : "▲ [실질적 해결 실행 조치 결여] 정중한 사과를 넘어 거래처를 실질적으로 안심시킬 수 있는 즉각적인 구제 조치(우선 송금, 전액 환불, 토큰 발급)를 제시해야 합니다.");
        } else if (topic === 'inquiry') {
          gaps.push(lang === 'jp' ? "▲ [受動的質問の回避] 単なる確認質問を避け、担当者として着手する即時調査および調整アクションを約束する主導的トーンを採用してください。" : lang === 'en' ? "▲ [Avoid Passive Interrogatives] Replace inquiries with active professional commitments to investigate logs and execute immediate adjustments." : "▲ [수동적 질의 지양] 단순 여부 문의 대신, 담당자로서 취할 즉각적인 전산 조회 및 정산 조율 액션을 확약하는 능동적 어조를 사용하세요.");
        } else if (len < 20) {
          gaps.push(lang === 'jp' ? "▲ [公式ビジネス書簡の作成推奨] 単文キーワードではなく、宛名、経緯説明、緊急措置、結びの言葉が揃った完成されたビジネス書簡として記述してください。" : lang === 'en' ? "▲ [Draft Complete Correspondence] Instead of single keywords, format your response as full professional correspondence with formal greetings, explanations, and sign-offs." : "▲ [공식 비즈니스 서신 서식 작성 권장] 단문 키워드 대신 수신인 호칭, 발생 경위 소명, 긴급 조치 내용, 맺음말이 완비된 전문 서신 형태로 작성해 보세요.");
        }

        // B. #1 Core Domain Priority Gap (from missing pillars)
        if (missingPillars.length > 0) {
          const rawG = missingPillars[0].gap[lang] || missingPillars[0].gap.kr;
          gaps.push(rawG.startsWith('▲') ? rawG : '▲ ' + rawG);
        }

        // C. Quantitative Turnaround SLA Timeline Gap
        if (precScoreRaw < 8 && gaps.length < 3) {
          gaps.push(lang === 'jp' ? "▲ [定量的SLA期限の提示] 取引先の不安を即座に解消する具体的な完了目標時刻（例：15分以内または本日15:00）を提示してください。" : lang === 'en' ? "▲ [Quantitative SLA Turnaround] State a concrete turnaround deadline (e.g. within 15 minutes or by 3:00 PM today) to restore counterparty certainty." : "▲ [정량적 SLA 기한 제시] 파트너사의 불안을 즉시 해소할 구체적 처리 마감 기한(예: 15분 이내 또는 금일 15:00)을 명시하세요.");
        }
      }

      // 8. Actionable Senior Coaching Directive
      let directive = "";
      if (simDomain === 'ptp') {
        if (lang === 'jp') {
          directive = totalScore >= 85 ? "【模範運用のポイント】照合済みPO額の先行送金と差額運賃の分離監査は、グローバルPTPにおける教科書的ベストプラクティスです。この調子でチーム内のSOPドキュメント化を推進してください。" : "【改善アクション】取引先の納品ストップを防ぐため、「確認済みPO額（RM 10,500）の即時先行送金」と「本日15:00までのクレジットノート発行要請」の2点を必ず明記してください。";
        } else if (lang === 'en') {
          directive = totalScore >= 85 ? "[Best Practice] Releasing approved PO funds while auditing freight variance separately is the gold standard of PTP finance. Continue enforcing this model across teams." : "[Action Directive] To stop supplier delivery holds, always explicitly pledge: 1) upfront release of verified PO funds (RM 10,500), and 2) a hard 3:00 PM turnaround for the Credit Note filing.";
        } else {
          directive = totalScore >= 85 ? "【모범 운영 포인트】확인된 PO 승인액 우선 결제와 차액 운임 분리 감사는 글로벌 PTP의 표준 베스트 프랙티스입니다. 조직 전체가 재사용 가능한 ERP 결제 시스템 자산으로 전파하세요." : "【개선 액션】공급업체 납품 중단을 방지하기 위해 '확인된 PO 승인 금액(RM 10,500) 우선 결제 집행'과 '금일 15:00까지 Credit Note 발행 완료'를 반드시 명문화하세요.";
        }
      } else if (simDomain === 'ota') {
        if (lang === 'jp') {
          directive = totalScore >= 85 ? "【模範運用のポイント】PMS連携ログ照合による顧客過失ゼロの証明と即時全額返金＋宿泊バウチャーは、重大クレームを生涯ロイヤルティへ転換する理想的なクローズドループです。" : "【改善アクション】旅行者のパニックを抑えるため、「PMSログ照合によるお客様過失ゼロの明言」と「手数料なし100%全額返金＋次回予約バウチャー」の組み合わせを明記してください。";
        } else if (lang === 'en') {
          directive = totalScore >= 85 ? "[Best Practice] Validating 0% traveler fault via PMS sync audit paired with instant refund and loyalty credits turns severe disruptions into high-CSAT brand retention." : "[Action Directive] Instantly reassure the traveler by citing: 1) PMS audit confirming 0% guest liability, and 2) immediate 100% refund with goodwill accommodation vouchers.";
        } else {
          directive = totalScore >= 85 ? "【모범 운영 포인트】PMS 시스템 로그 대조로 고객 과실 0%를 확정하고, 즉시 전액 환불과 차회 바우처를 결합한 조치는 고객 이탈을 완벽히 방어하는 시니어 CX의 정석입니다." : "【개선 액션】여행객의 불안을 종식시키기 위해 'PMS 연동 로그 대조를 통한 고객 과실 0% 공인' 및 '100% 전액 환불과 차기 숙박 보상 바우처 지급'을 구체적으로 약속하세요.";
        }
      } else if (simDomain === 'chargeback') {
        if (lang === 'jp') {
          directive = totalScore >= 85 ? "【模範運用のポイント】3Dセキュア2.0の責任転換規則を援用し、署名受領書で24時間以内に異議申立を行うことで、加盟店の運転資金保留を早期解除する理想的なリスク防御です。" : "【改善アクション】加盟店の資金繰り不安を解消するため、「3Dセキュア2.0の責任転換条項」と「署名付き配達完了証（POD）による24時間以内の公式異議申立」を明記してください。";
        } else if (lang === 'en') {
          directive = totalScore >= 85 ? "[Best Practice] Invoking 3DS 2.0 liability shift rules backed by signed POD within 24h bank representment safeguards merchant working capital and eliminates partner churn." : "[Action Directive] Quell merchant anxiety by citing: 1) card brand 3DS 2.0 liability shift protections, and 2) filing representment within 24h backed by signed delivery receipts.";
        } else {
          directive = totalScore >= 85 ? "【모범 운영 포인트】3D Secure 2.0 책임전환 조항을 소명하고, 배송완료 서명 증빙으로 24시간 내 이의제기를 단행하여 가맹점 정산 유동성을 수호하는 완벽한 핀테크 리스크 대응입니다." : "【개선 액션】가맹점 이탈을 막기 위해 '3D Secure 2.0 책임전환(Liability Shift) 규정 소명'과 '서명 배송완료 증빙(POD) 기반 24시간 내 이의제기(Representment)'를 명확히 제시하세요.";
        }
      } else {
        if (lang === 'jp') {
          directive = totalScore >= 85 ? "【模範運用のポイント】SAML緊急トークンの即時プロビジョニングによる15分以内復旧と、誤請求額のインボイス相殺クレジットは、エンタープライズSaaSの極致と言える統制力です。" : "【改善アクション】企業のIT統括責任者を安堵させるため、「15分以内のSAML代替トークン自動プロビジョニング」と「監査ログに基づく$4,500全額のインボイス相殺」を明記してください。";
        } else if (lang === 'en') {
          directive = totalScore >= 85 ? "[Best Practice] Restoring 300 users in 15 mins via emergency SAML token provisioning alongside transparent $4,500 billing ledger credits proves elite B2B technical governance." : "[Action Directive] Defuse enterprise executive escalations by confirming: 1) 15-minute emergency fallback token provisioning, and 2) audit log reconciliation crediting the $4,500 overage.";
        } else {
          directive = totalScore >= 85 ? "【모범 운영 포인트】15분 SLA 약정 준수를 위한 비상 SAML 대체 토큰 프로비저닝과 감사 로그 대조 기반 $4,500 전액 상계 처리는 B2B 엔터프라이즈 기술 운영의 정점입니다." : "【개선 액션】기업 IT 총괄 관리자의 항의를 방어하기 위해 '15분 이내 비상 임시 토큰 프로비저닝 복구'와 '감사 로그 대조를 통한 $4,500 전액 익월 인보이스 상계'를 명시하세요.";
        }
      }
      setSimResult({
        score: totalScore,
        grade,
        diagnosisBadge,
        complianceRate,
        toneScore,
        riskScore,
        matched,
        missing,
        summary,
        strengths,
        gaps,
        directive,
        feedback: summary
      });
      setIsEvaluating(false);
      showToast(lang === 'jp' ? 'AI実務適合性分析が完了しました！' : lang === 'en' ? 'AI Competency Analysis Complete!' : '정밀 AI 직무 적합성 분석 완료!');
    }, 500);
  };
  const filteredDeliverables = deliverables.filter(d => {
    if (filterCategory === 'all') return true;
    return d.lang.includes(filterCategory);
  });
  return /*#__PURE__*/React.createElement("div", {
    className: "min-h-screen bg-slate-950 text-slate-100 font-sans pb-20"
  }, notification && /*#__PURE__*/React.createElement("div", {
    className: "fixed top-5 right-5 z-50 bg-indigo-900 text-white px-4 py-3 rounded-xl shadow-2xl border border-indigo-500/40 flex items-center space-x-2 animate-bounce"
  }, /*#__PURE__*/React.createElement(SparklesIcon, null), /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-bold"
  }, notification)), /*#__PURE__*/React.createElement("div", {
    className: "language-selector fixed top-4 right-4 z-40 px-3 py-2 rounded-xl flex items-center gap-2 border"
  }, /*#__PURE__*/React.createElement("label", {
    htmlFor: "languageSelect",
    className: "text-xs font-bold text-slate-300 flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-globe text-cyan-400"
  }), /*#__PURE__*/React.createElement("span", {
    className: "hidden sm:inline"
  }, curT.langPriorityLabel)), /*#__PURE__*/React.createElement("select", {
    id: "languageSelect",
    value: lang,
    onChange: e => {
      setLang(e.target.value);
      showToast(`Language switched to ${e.target.value.toUpperCase()}`);
    },
    className: "text-xs font-black px-2.5 py-1 rounded-lg focus:outline-none focus:ring-1 focus:ring-cyan-400 cursor-pointer"
  }, /*#__PURE__*/React.createElement("option", {
    value: "kr"
  }, "KR (\uD55C\uAD6D\uC5B4)"), /*#__PURE__*/React.createElement("option", {
    value: "en"
  }, "EN (English)"), /*#__PURE__*/React.createElement("option", {
    value: "jp"
  }, "JP (\u65E5\u672C\u8A9E)"))), /*#__PURE__*/React.createElement("header", {
    className: "border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-30 px-6 py-3.5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "max-w-7xl mx-auto flex justify-between items-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-black text-sm tracking-wider text-white"
  }, "DAVID YEONWOO PARK"), /*#__PURE__*/React.createElement("span", {
    className: "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-[10px] font-bold px-2 py-0.5 rounded-full"
  }, "SENIOR OPS ARCHITECT")), /*#__PURE__*/React.createElement("nav", {
    className: "hidden md:flex items-center gap-6 text-xs text-slate-400 font-semibold pr-36"
  }, /*#__PURE__*/React.createElement("a", {
    href: "#profile",
    className: "hover:text-cyan-400 transition-colors"
  }, "Profile"), /*#__PURE__*/React.createElement("a", {
    href: "#work",
    className: "hover:text-cyan-400 transition-colors"
  }, "Selected Work (3)"), /*#__PURE__*/React.createElement("a", {
    href: "#development",
    className: "hover:text-cyan-400 transition-colors"
  }, "Development Map"), /*#__PURE__*/React.createElement("a", {
    href: "#deliverables",
    className: "hover:text-cyan-400 transition-colors"
  }, "Deliverables (16)"), /*#__PURE__*/React.createElement("a", {
    href: "#simulator",
    className: "hover:text-cyan-400 transition-colors"
  }, "BPO Simulator")))), /*#__PURE__*/React.createElement("div", {
    className: "max-w-7xl mx-auto px-6 pt-8 space-y-12"
  }, /*#__PURE__*/React.createElement("section", {
    id: "profile",
    className: "bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 rounded-3xl p-6 md:p-8 shadow-2xl"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col md:flex-row items-center gap-6 md:gap-8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "relative group"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-28 h-28 md:w-32 md:h-32 rounded-full border-4 border-cyan-400 p-0.5 bg-slate-900 shadow-2xl shadow-cyan-500/30 overflow-hidden flex items-center justify-center ring-2 ring-cyan-500/40"
  }, /*#__PURE__*/React.createElement("img", {
    src: "./\uC2A4\uD06C\uB9B0\uC0F7 2025-10-20 202241.png",
    alt: "\uBC15\uC5F0\uC6B0 (David Yeonwoo Park)",
    className: "w-full h-full object-cover object-top rounded-full transition-transform duration-300 group-hover:scale-105",
    onError: e => {
      if (typeof userProfilePhoto !== 'undefined' && e.currentTarget.src !== userProfilePhoto) {
        e.currentTarget.src = userProfilePhoto;
      } else {
        e.currentTarget.style.display = 'none';
        if (e.currentTarget.nextElementSibling) {
          e.currentTarget.nextElementSibling.style.display = 'flex';
        }
      }
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "hidden w-full h-full items-center justify-center text-3xl font-black text-cyan-400"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-user-tie"
  }))), /*#__PURE__*/React.createElement("span", {
    className: "absolute bottom-1 right-1 w-5 h-5 bg-emerald-500 border-2 border-slate-950 rounded-full shadow-md",
    title: "Active & Ready for Hire"
  })), /*#__PURE__*/React.createElement("div", {
    className: "text-center md:text-left flex-1 space-y-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap items-center justify-center md:justify-start gap-2.5"
  }, /*#__PURE__*/React.createElement("h1", {
    className: "text-2xl md:text-3xl font-black text-white"
  }, "\uBC15\uC5F0\uC6B0 (David Yeonwoo Park)"), /*#__PURE__*/React.createElement("span", {
    className: "text-xs bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-bold px-2.5 py-0.5 rounded-full"
  }, "8.5y Global Ops")), /*#__PURE__*/React.createElement("h2", {
    className: "text-sm md:text-base font-bold text-cyan-400"
  }, curT.profileRole), /*#__PURE__*/React.createElement("p", {
    className: "text-xs md:text-sm text-slate-300 leading-relaxed max-w-3xl"
  }, curT.profileDesc), /*#__PURE__*/React.createElement("div", {
    className: "pt-3 flex flex-wrap justify-center md:justify-start gap-3"
  }, /*#__PURE__*/React.createElement("a", {
    href: "Cover letter.pdf",
    target: "_blank",
    className: "bg-red-600 hover:bg-red-500 text-white font-bold text-xs py-2 px-4 rounded-xl shadow-lg transition-all inline-flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-file-pdf"
  }), /*#__PURE__*/React.createElement("span", null, curT.viewCoverLetter)), /*#__PURE__*/React.createElement("button", {
    onClick: () => setContactModalOpen(true),
    className: "bg-slate-800 hover:bg-slate-700 text-white font-bold text-xs py-2 px-4 rounded-xl border border-slate-700 shadow-lg transition-all inline-flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-address-book text-cyan-400"
  }), /*#__PURE__*/React.createElement("span", null, curT.contactMe)), /*#__PURE__*/React.createElement("a", {
    href: "https://www.linkedin.com/in/yeonwoo-park-b54bba356/",
    target: "_blank",
    className: "bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs py-2 px-4 rounded-xl shadow-lg transition-all inline-flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-brands fa-linkedin"
  }), /*#__PURE__*/React.createElement("span", null, "LinkedIn")), /*#__PURE__*/React.createElement("a", {
    href: "https://github.com/Salmonyeonwoo",
    target: "_blank",
    className: "bg-slate-900 hover:bg-slate-800 text-white font-bold text-xs py-2 px-4 rounded-xl border border-slate-700 shadow-lg transition-all inline-flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-brands fa-github"
  }), /*#__PURE__*/React.createElement("span", null, "GitHub")))))), /*#__PURE__*/React.createElement("section", {
    className: "grid grid-cols-1 lg:grid-cols-12 gap-8 items-center pt-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "lg:col-span-7 space-y-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "inline-flex items-center gap-2 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-3 py-1 rounded-full text-xs font-bold"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-2 h-2 rounded-full bg-cyan-400 animate-pulse"
  }), /*#__PURE__*/React.createElement("span", null, curT.heroEyebrow)), /*#__PURE__*/React.createElement("h1", {
    className: "text-3xl sm:text-5xl lg:text-6xl font-black text-white leading-tight tracking-tight"
  }, curT.heroTitle1, /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    className: "text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-indigo-400"
  }, curT.heroTitle2)), /*#__PURE__*/React.createElement("div", {
    className: "bg-gradient-to-r from-slate-900 via-indigo-950/60 to-slate-900 border-l-4 border-cyan-400 border-y border-r border-slate-800 p-4 sm:p-5 rounded-2xl shadow-xl"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-cyan-400 uppercase tracking-wider block mb-1"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-compass mr-1.5"
  }), curT.philosophyTag), /*#__PURE__*/React.createElement("p", {
    className: "text-sm sm:text-base font-bold text-white leading-relaxed"
  }, curT.philosophyQuote), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 mt-2"
  }, curT.philosophySub)), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-3 pt-2"
  }, /*#__PURE__*/React.createElement("a", {
    href: "#work",
    className: "bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-black text-xs px-5 py-3 rounded-xl shadow-lg shadow-cyan-500/20 transition-all inline-flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", null, curT.viewProjectsBtn)), /*#__PURE__*/React.createElement("a", {
    href: "#deliverables",
    className: "bg-slate-900 hover:bg-slate-800 text-white font-bold text-xs px-5 py-3 rounded-xl border border-slate-800 transition-all inline-flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-folder-open text-indigo-400"
  }), /*#__PURE__*/React.createElement("span", null, curT.viewDocsBtn)), /*#__PURE__*/React.createElement("a", {
    href: "#simulator",
    className: "bg-slate-900 hover:bg-slate-800 text-white font-bold text-xs px-5 py-3 rounded-xl border border-slate-800 transition-all inline-flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-headset text-emerald-400"
  }), /*#__PURE__*/React.createElement("span", null, curT.viewSimBtn)))), /*#__PURE__*/React.createElement("div", {
    className: "lg:col-span-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-gradient-to-b from-slate-900 to-indigo-950/70 border border-slate-800 rounded-3xl p-6 shadow-2xl space-y-3 relative overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-400 via-indigo-500 to-emerald-400"
  }), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center border-b border-slate-800 pb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-400 tracking-wider uppercase"
  }, curT.snapHead), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded-full font-bold"
  }, "10s Recruiter Match")), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2.5 text-xs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center py-1 border-b border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.snapK1), /*#__PURE__*/React.createElement("strong", {
    className: "text-cyan-300 font-bold"
  }, curT.snapV1)), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center py-1 border-b border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.snapK2), /*#__PURE__*/React.createElement("strong", {
    className: "text-indigo-300 font-bold"
  }, curT.snapV2)), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center py-1 border-b border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.snapK3), /*#__PURE__*/React.createElement("strong", {
    className: "text-white font-bold"
  }, curT.snapV3)), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center py-1 border-b border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.snapK4), /*#__PURE__*/React.createElement("strong", {
    className: "text-emerald-300 font-bold"
  }, curT.snapV4)), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center py-1 border-b border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.snapK5), /*#__PURE__*/React.createElement("strong", {
    className: "text-amber-300 font-bold"
  }, curT.snapV5)), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center py-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.snapK6), /*#__PURE__*/React.createElement("strong", {
    className: "text-teal-300 font-bold"
  }, curT.snapV6)))))), /*#__PURE__*/React.createElement("section", {
    className: "grid grid-cols-2 md:grid-cols-4 gap-4 py-4 border-y border-slate-800/80 bg-slate-900/30 rounded-2xl px-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-center md:border-r border-slate-800 py-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-2xl sm:text-3xl font-black text-white"
  }, "8.5", /*#__PURE__*/React.createElement("span", {
    className: "text-cyan-400 text-lg"
  }, "y")), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] text-slate-400 block font-semibold mt-0.5"
  }, curT.m1)), /*#__PURE__*/React.createElement("div", {
    className: "text-center md:border-r border-slate-800 py-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-2xl sm:text-3xl font-black text-white"
  }, "3", /*#__PURE__*/React.createElement("span", {
    className: "text-indigo-400 text-lg"
  }, " Lang")), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] text-slate-400 block font-semibold mt-0.5"
  }, curT.m2)), /*#__PURE__*/React.createElement("div", {
    className: "text-center md:border-r border-slate-800 py-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-2xl sm:text-3xl font-black text-white"
  }, "4", /*#__PURE__*/React.createElement("span", {
    className: "text-emerald-400 text-lg"
  }, " BPO Domains")), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] text-slate-400 block font-semibold mt-0.5"
  }, curT.m3)), /*#__PURE__*/React.createElement("div", {
    className: "text-center py-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-2xl sm:text-3xl font-black text-cyan-400"
  }, "01"), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] text-slate-400 block font-semibold mt-0.5"
  }, curT.m4))), /*#__PURE__*/React.createElement("section", {
    id: "work",
    className: "space-y-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col md:flex-row justify-between items-start md:items-end gap-2 border-b border-slate-800 pb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-400 uppercase tracking-wider block"
  }, curT.workEyebrow), /*#__PURE__*/React.createElement("h2", {
    className: "text-2xl sm:text-3xl font-black text-white mt-1"
  }, curT.workTitle)), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400"
  }, curT.workHint)), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 md:grid-cols-3 gap-5"
  }, ['ptp', 'recovery', 'dashboard'].map((projKey, index) => {
    const proj = projectData[projKey];
    const isSelected = selectedProject === projKey;
    const badgeColor = index === 0 ? 'text-indigo-300 bg-indigo-950 border-indigo-500/30' : index === 1 ? 'text-emerald-300 bg-emerald-950 border-emerald-500/30' : 'text-cyan-300 bg-cyan-950 border-cyan-500/30';
    const borderAccent = index === 0 ? 'text-cyan-400' : index === 1 ? 'text-emerald-400' : 'text-blue-400';
    return /*#__PURE__*/React.createElement("div", {
      key: projKey,
      onClick: () => setSelectedProject(projKey),
      className: `p-6 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between space-y-4 ${isSelected ? 'bg-slate-900/90 border-cyan-400 shadow-xl shadow-cyan-500/10 ring-1 ring-cyan-400' : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'}`
    }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
      className: "flex justify-between items-center mb-3"
    }, /*#__PURE__*/React.createElement("span", {
      className: `text-[11px] font-black tracking-wider ${borderAccent}`
    }, proj.kicker[lang] || proj.kicker.kr), /*#__PURE__*/React.createElement("span", {
      className: `text-[10px] px-2.5 py-0.5 rounded-full font-bold border ${badgeColor}`
    }, projKey === 'ptp' ? '3-Way Match' : projKey === 'recovery' ? 'CX Recovery' : 'Scalable Ops')), /*#__PURE__*/React.createElement("h3", {
      className: "text-lg font-black text-white"
    }, proj.title[lang] || proj.title.kr), /*#__PURE__*/React.createElement("p", {
      className: "text-xs text-slate-400 mt-2 leading-relaxed"
    }, proj.shortDesc[lang] || proj.shortDesc.kr)), /*#__PURE__*/React.createElement("div", {
      className: `text-xs font-bold flex items-center gap-1.5 pt-2 ${borderAccent}`
    }, /*#__PURE__*/React.createElement("span", null, curT.clickToInspect), /*#__PURE__*/React.createElement("i", {
      className: "fa-solid fa-arrow-right"
    })));
  })), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900/90 border border-slate-800 rounded-3xl p-6 md:p-8 shadow-2xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch"
  }, /*#__PURE__*/React.createElement("div", {
    className: "lg:col-span-6 space-y-5 flex flex-col justify-between"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-cyan-400 uppercase tracking-wider block"
  }, projectData[selectedProject].kicker[lang] || projectData[selectedProject].kicker.kr), /*#__PURE__*/React.createElement("h3", {
    className: "text-2xl font-black text-white mt-1"
  }, projectData[selectedProject].title[lang] || projectData[selectedProject].title.kr), /*#__PURE__*/React.createElement("p", {
    className: "text-xs md:text-sm text-slate-300 mt-2 leading-relaxed"
  }, projectData[selectedProject].desc[lang] || projectData[selectedProject].desc.kr)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-slate-400 uppercase tracking-wider block mb-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-bars-progress mr-1.5 text-cyan-400"
  }), curT.flowHeading), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 sm:grid-cols-4 gap-2 text-center"
  }, projectData[selectedProject].steps.map((step, idx) => /*#__PURE__*/React.createElement("div", {
    key: idx,
    className: "bg-slate-950 p-2.5 rounded-xl border border-slate-800"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-black text-cyan-400 block"
  }, step.num), /*#__PURE__*/React.createElement("b", {
    className: "text-xs font-bold text-white block mt-0.5 leading-snug"
  }, step.title[lang] || step.title.kr), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-slate-400 block mt-0.5"
  }, step.sub[lang] || step.sub.kr))))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs pt-1"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-3.5 rounded-xl border-l-4 border-rose-500 border-y border-r border-slate-800/80"
  }, /*#__PURE__*/React.createElement("b", {
    className: "text-rose-400 font-bold block mb-1"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-triangle-exclamation mr-1"
  }), " ", curT.lblProblem), /*#__PURE__*/React.createElement("span", {
    className: "text-slate-300 leading-relaxed block"
  }, projectData[selectedProject].problem[lang] || projectData[selectedProject].problem.kr)), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-3.5 rounded-xl border-l-4 border-blue-500 border-y border-r border-slate-800/80"
  }, /*#__PURE__*/React.createElement("b", {
    className: "text-blue-400 font-bold block mb-1"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-scale-balanced mr-1"
  }), " ", curT.lblCriteria), /*#__PURE__*/React.createElement("span", {
    className: "text-slate-300 leading-relaxed block"
  }, projectData[selectedProject].criteria[lang] || projectData[selectedProject].criteria.kr)), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-3.5 rounded-xl border-l-4 border-amber-500 border-y border-r border-slate-800/80"
  }, /*#__PURE__*/React.createElement("b", {
    className: "text-amber-400 font-bold block mb-1"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-gears mr-1"
  }), " ", curT.lblWorkflow), /*#__PURE__*/React.createElement("span", {
    className: "text-slate-300 leading-relaxed block"
  }, projectData[selectedProject].workflow[lang] || projectData[selectedProject].workflow.kr)), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-3.5 rounded-xl border-l-4 border-emerald-500 border-y border-r border-slate-800/80"
  }, /*#__PURE__*/React.createElement("b", {
    className: "text-emerald-400 font-bold block mb-1"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-award mr-1"
  }), " ", curT.lblImpact), /*#__PURE__*/React.createElement("span", {
    className: "text-slate-300 leading-relaxed block"
  }, projectData[selectedProject].impact[lang] || projectData[selectedProject].impact.kr)))), /*#__PURE__*/React.createElement("div", {
    className: "lg:col-span-6 flex flex-col justify-center"
  }, selectedProject === 'ptp' && /*#__PURE__*/React.createElement(PtpVisualDiagram, {
    lang: lang
  }), selectedProject === 'recovery' && /*#__PURE__*/React.createElement(AgodaVisualDiagram, {
    lang: lang
  }), selectedProject === 'dashboard' && /*#__PURE__*/React.createElement(DashboardVisualDiagram, {
    lang: lang
  })))), /*#__PURE__*/React.createElement("section", {
    id: "development",
    className: "space-y-6 pt-6 border-t border-slate-800"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-400 uppercase tracking-wider block"
  }, curT.devEyebrow), /*#__PURE__*/React.createElement("h2", {
    className: "text-2xl sm:text-3xl font-black text-white mt-1"
  }, curT.devTitle), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 mt-1"
  }, curT.devSubtitle)), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 md:grid-cols-3 gap-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900/70 border border-slate-800 p-6 rounded-2xl relative overflow-hidden space-y-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 left-0 right-0 h-1 bg-cyan-400"
  }), /*#__PURE__*/React.createElement("div", {
    className: "w-10 h-10 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center text-cyan-400 font-black"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-file-shield"
  })), /*#__PURE__*/React.createElement("h3", {
    className: "text-base font-black text-white"
  }, curT.dev1Title), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 leading-relaxed"
  }, curT.dev1Desc)), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900/70 border border-slate-800 p-6 rounded-2xl relative overflow-hidden space-y-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 left-0 right-0 h-1 bg-indigo-500"
  }), /*#__PURE__*/React.createElement("div", {
    className: "w-10 h-10 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center text-indigo-400 font-black"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-diagram-project"
  })), /*#__PURE__*/React.createElement("h3", {
    className: "text-base font-black text-white"
  }, curT.dev2Title), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 leading-relaxed"
  }, curT.dev2Desc)), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900/70 border border-slate-800 p-6 rounded-2xl relative overflow-hidden space-y-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 left-0 right-0 h-1 bg-emerald-400"
  }), /*#__PURE__*/React.createElement("div", {
    className: "w-10 h-10 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center text-emerald-400 font-black"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-recycle"
  })), /*#__PURE__*/React.createElement("h3", {
    className: "text-base font-black text-white"
  }, curT.dev3Title), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 leading-relaxed"
  }, curT.dev3Desc)))), /*#__PURE__*/React.createElement("section", {
    id: "deliverables",
    className: "space-y-6 pt-6 border-t border-slate-800"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col md:flex-row justify-between items-start md:items-end gap-2"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-400 uppercase tracking-wider block"
  }, curT.docsEyebrow), /*#__PURE__*/React.createElement("h2", {
    className: "text-2xl sm:text-3xl font-black text-white mt-1"
  }, curT.docsTitle), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 mt-1"
  }, curT.docsSubtitle)), /*#__PURE__*/React.createElement("div", {
    className: "flex gap-2"
  }, ['all', 'kr', 'en', 'jp'].map(cat => /*#__PURE__*/React.createElement("button", {
    key: cat,
    onClick: () => setFilterCategory(cat),
    className: `px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${filterCategory === cat ? 'bg-cyan-500 text-slate-950 font-black shadow-md' : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'}`
  }, cat === 'all' ? lang === 'jp' ? 'すべて (16)' : lang === 'en' ? 'All (16)' : '전체 (16)' : cat === 'kr' ? '🇰🇷 KR' : cat === 'en' ? '🇺🇸 EN' : '🇯🇵 JP')))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
  }, filteredDeliverables.map(doc => /*#__PURE__*/React.createElement("a", {
    key: doc.id,
    href: doc.file,
    target: "_blank",
    className: "bg-slate-900/80 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 p-5 rounded-2xl transition-all shadow-sm flex flex-col justify-between space-y-3 group"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-black px-2 py-0.5 rounded-md bg-slate-950 text-indigo-400 border border-slate-800"
  }, doc.badge), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-cyan-400 uppercase"
  }, doc.lang.toUpperCase())), /*#__PURE__*/React.createElement("h4", {
    className: "text-sm font-bold text-white group-hover:text-cyan-300 transition-colors"
  }, doc.title[lang] || doc.title.kr), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 mt-1 line-clamp-2"
  }, doc.desc[lang] || doc.desc.kr)), /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] font-bold text-cyan-400 flex items-center gap-1"
  }, /*#__PURE__*/React.createElement("span", null, curT.openDocBtn), /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-arrow-up-right-from-square text-[10px]"
  })))))), /*#__PURE__*/React.createElement("section", {
    id: "simulator",
    className: "space-y-6 pt-6 border-t border-slate-800"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-400 uppercase tracking-wider block"
  }, curT.simEyebrow), /*#__PURE__*/React.createElement("h2", {
    className: "text-2xl sm:text-3xl font-black text-white mt-1"
  }, curT.simTitle), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 mt-1"
  }, curT.simDesc)), /*#__PURE__*/React.createElement("div", {
    className: "bg-gradient-to-r from-slate-900 via-indigo-950/30 to-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl space-y-6"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-slate-400 uppercase tracking-wider block mb-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-layer-group text-cyan-400 mr-1.5"
  }), lang === 'jp' ? 'BPO現場・顧客対応ドメインを選択' : lang === 'en' ? 'Select BPO Operations / Customer Support Domain' : '고객 응대 및 BPO 부서 분야 선택'), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5"
  }, Object.keys(bpoDomains).map(domainKey => {
    const dom = bpoDomains[domainKey];
    const isCurrent = simDomain === domainKey;
    return /*#__PURE__*/React.createElement("button", {
      key: domainKey,
      onClick: () => {
        setSimDomain(domainKey);
        setUserAnswer('');
        setSimResult(null);
      },
      className: `px-3.5 py-2.5 rounded-xl text-xs font-bold transition-all text-left flex flex-col justify-between border ${isCurrent ? 'bg-indigo-600 text-white border-indigo-400 shadow-lg shadow-indigo-500/20 ring-1 ring-indigo-400' : 'bg-slate-950 text-slate-400 hover:text-white border-slate-800 hover:bg-slate-900'}`
    }, /*#__PURE__*/React.createElement("span", {
      className: "text-[10px] uppercase font-mono tracking-wider text-cyan-300 block mb-1"
    }, dom.badge), /*#__PURE__*/React.createElement("span", {
      className: "font-extrabold text-xs block leading-snug"
    }, dom.tabLabel[lang] || dom.tabLabel.kr));
  }))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch"
  }, /*#__PURE__*/React.createElement("div", {
    className: "lg:col-span-7 space-y-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-4 rounded-xl border border-cyan-500/30 text-xs space-y-1.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-cyan-400 uppercase tracking-wider block"
  }, curT.policyLabel), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-300 leading-relaxed font-medium"
  }, bpoDomains[simDomain].policy[lang] || bpoDomains[simDomain].policy.kr)), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs text-slate-300"
  }, /*#__PURE__*/React.createElement("strong", {
    className: "text-indigo-400 block mb-1"
  }, curT.situationLabel), bpoDomains[simDomain].context[lang] || bpoDomains[simDomain].context.kr), /*#__PURE__*/React.createElement("div", {
    className: "bg-indigo-950/50 p-4 rounded-xl border border-indigo-500/30"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-indigo-300 uppercase block mb-1"
  }, curT.queryLabel), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-200 font-mono italic leading-relaxed bg-slate-950/80 p-3 rounded-lg border border-slate-800"
  }, bpoDomains[simDomain].query)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-bold text-white flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-pen-nib text-cyan-400"
  }), curT.draftLabel), /*#__PURE__*/React.createElement("button", {
    onClick: loadSimSample,
    className: "text-[11px] text-cyan-400 hover:text-cyan-300 font-bold underline transition-colors"
  }, curT.autoLoadBtn)), /*#__PURE__*/React.createElement("textarea", {
    value: userAnswer,
    onChange: e => setUserAnswer(e.target.value),
    rows: "6",
    placeholder: lang === 'jp' ? '該当分野のポリシーに則り、事実関係の確認、先行是正措置、タイムラインを明示したプロフェッショナルな実務対応文を作成してください...' : lang === 'en' ? 'Draft your business resolution referencing operational policies, timeline commitments, and corrective actions...' : '해당 분야의 규정과 절차에 맞추어 선제 조치, 시스템 대조 사실, 명확한 해결 타임라인을 비즈니스 톤으로 작성하세요...',
    className: "w-full bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-400 leading-relaxed"
  }), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mt-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] text-slate-400"
  }, lang === 'jp' ? '推奨キーワード:' : lang === 'en' ? 'Recommended Keywords:' : '권장 키워드:', " ", /*#__PURE__*/React.createElement("strong", {
    className: "text-cyan-400 font-mono"
  }, (bpoDomains[simDomain].keywords && (bpoDomains[simDomain].keywords[lang] || bpoDomains[simDomain].keywords.kr) || []).slice(0, 5).join(', '))), /*#__PURE__*/React.createElement("button", {
    onClick: evaluateAnswer,
    disabled: isEvaluating,
    className: "bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-lg transition-all flex items-center gap-2"
  }, isEvaluating ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"
  }), /*#__PURE__*/React.createElement("span", null, lang === 'jp' ? 'AI分析中...' : lang === 'en' ? 'Evaluating...' : 'AI 분석 중...')) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SparklesIcon, null), /*#__PURE__*/React.createElement("span", null, curT.evalBtn)))))), /*#__PURE__*/React.createElement("div", {
    className: "lg:col-span-5 flex flex-col justify-between bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between items-center border-b border-slate-800 pb-3 mb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-xs font-black text-cyan-400 uppercase tracking-wider"
  }, curT.evalReportHead), /*#__PURE__*/React.createElement("span", {
    className: `text-[10px] font-black px-2.5 py-0.5 rounded-full border ${simResult ? 'bg-emerald-950 text-emerald-300 border-emerald-500/30' : 'bg-slate-900 text-slate-500 border-slate-800'}`
  }, simResult ? simResult.grade : 'READY')), simResult ? /*#__PURE__*/React.createElement("div", {
    className: "space-y-4 animate-in fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 gap-3 text-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 p-3 rounded-xl border border-slate-800"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-slate-400 uppercase font-bold block"
  }, curT.evalScoreLabel), /*#__PURE__*/React.createElement("span", {
    className: "text-2xl font-black text-cyan-400 mt-0.5 block"
  }, simResult.score, " / 100")), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 p-3 rounded-xl border border-slate-800"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-slate-400 uppercase font-bold block"
  }, curT.evalMatchedLabel), /*#__PURE__*/React.createElement("span", {
    className: "text-2xl font-black text-emerald-400 mt-0.5 block"
  }, simResult.matched.length, " ", lang === 'jp' ? '個' : lang === 'en' ? 'Hits' : '개'))), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 p-3.5 rounded-xl border border-slate-800 space-y-2.5 text-xs"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-black text-slate-300 uppercase tracking-wider block"
  }, curT.evalSubMetrics), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between text-[11px] mb-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.subCompliance), /*#__PURE__*/React.createElement("span", {
    className: "text-cyan-300 font-bold"
  }, simResult.complianceRate, "%")), /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-slate-950 h-2 rounded-full overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-cyan-400 h-full rounded-full transition-all",
    style: {
      width: `${simResult.complianceRate}%`
    }
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between text-[11px] mb-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.subTone), /*#__PURE__*/React.createElement("span", {
    className: "text-indigo-300 font-bold"
  }, simResult.toneScore, "%")), /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-slate-950 h-2 rounded-full overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-indigo-500 h-full rounded-full transition-all",
    style: {
      width: `${simResult.toneScore}%`
    }
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between text-[11px] mb-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-slate-400"
  }, curT.subRisk), /*#__PURE__*/React.createElement("span", {
    className: "text-emerald-300 font-bold"
  }, simResult.riskScore, "%")), /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-slate-950 h-2 rounded-full overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-emerald-400 h-full rounded-full transition-all",
    style: {
      width: `${simResult.riskScore}%`
    }
  })))), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2 text-xs"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-emerald-400 font-bold uppercase block mb-1"
  }, "\u2713 ", curT.evalMatchedLabel, ":"), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-1.5"
  }, simResult.matched.length > 0 ? simResult.matched.map((kw, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    className: "bg-emerald-950 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold"
  }, kw)) : /*#__PURE__*/React.createElement("span", {
    className: "text-slate-500 text-[10px]"
  }, lang === 'jp' ? 'なし' : lang === 'en' ? 'None' : '없음'))), simResult.missing.length > 0 && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-amber-400 font-bold uppercase block mb-1"
  }, "\u25B2 ", curT.evalMissingLabel, ":"), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-1.5"
  }, simResult.missing.map((kw, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    className: "bg-amber-950 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded-md text-[10px] font-mono"
  }, kw))))), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900/95 p-4 rounded-xl border border-slate-800 space-y-3.5 text-xs shadow-lg"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between border-b border-slate-800/80 pb-2"
  }, /*#__PURE__*/React.createElement("strong", {
    className: "text-cyan-400 flex items-center gap-1.5 text-xs font-bold"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-brain"
  }), curT.evalFeedbackLabel, ":"), /*#__PURE__*/React.createElement("span", {
    className: `text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${simResult.score >= 90 ? 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40' : simResult.score >= 75 ? 'bg-indigo-950/80 text-indigo-300 border-indigo-500/40' : simResult.score >= 50 ? 'bg-amber-950/80 text-amber-300 border-amber-500/40' : 'bg-rose-950/80 text-rose-300 border-rose-500/40'}`
  }, simResult.diagnosisBadge)), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-200 font-medium leading-relaxed"
  }, simResult.summary), simResult.strengths && simResult.strengths.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "space-y-1.5 pt-2 border-t border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-black text-emerald-400 uppercase tracking-wider block"
  }, curT.evalStrengthsLabel || '✓ Verified Strengths', " (", simResult.strengths.length, ")"), /*#__PURE__*/React.createElement("div", {
    className: "space-y-1"
  }, simResult.strengths.map((st, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "flex items-start gap-1.5 text-[11px] text-slate-300"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-check text-emerald-400 text-[10px] mt-0.5 shrink-0"
  }), /*#__PURE__*/React.createElement("span", null, st))))), simResult.gaps && simResult.gaps.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "space-y-1.5 pt-2 border-t border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-black text-amber-400 uppercase tracking-wider block"
  }, curT.evalGapsLabel || '▲ Remediation Gaps', " (", simResult.gaps.length, ")"), /*#__PURE__*/React.createElement("div", {
    className: "space-y-1"
  }, simResult.gaps.map((gp, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "flex items-start gap-1.5 text-[11px] text-slate-300"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-triangle-exclamation text-amber-400 text-[10px] mt-0.5 shrink-0"
  }), /*#__PURE__*/React.createElement("span", null, gp))))), /*#__PURE__*/React.createElement("div", {
    className: "bg-indigo-950/40 border border-indigo-500/30 p-3 rounded-xl space-y-1 mt-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-black text-cyan-300 uppercase block flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-compass"
  }), curT.evalDirectiveLabel || 'Senior Operational Directive'), /*#__PURE__*/React.createElement("p", {
    className: "text-[11px] text-slate-300 font-medium leading-relaxed"
  }, simResult.directive)))) : /*#__PURE__*/React.createElement("div", {
    className: "text-center py-12 text-slate-500 space-y-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-clipboard-check text-3xl"
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-xs"
  }, lang === 'jp' ? '左側のエディタで実務文を作成後、「AI適合性評価を実行」ボタンを押すと、多角的なルーブリック分析レポートが生成されます。' : lang === 'en' ? 'Draft your response on the left and click "Run AI Evaluation" to generate comprehensive rubric feedback.' : '좌측 폼에 실무 대응문을 작성 후 [실무 적합성 평가 실행] 버튼을 클릭하면 도메인별 다면 평가 결과가 생성됩니다.'))), /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] text-slate-400 pt-2 border-t border-slate-800/80"
  }, curT.evalFooterNote)))))), /*#__PURE__*/React.createElement("footer", {
    className: "max-w-7xl mx-auto px-6 mt-16 text-center text-xs text-slate-500 border-t border-slate-800 pt-6 space-y-1"
  }, /*#__PURE__*/React.createElement("p", null, "\xA9 2026 David Yeonwoo Park Career Development Suite. All rights reserved."), /*#__PURE__*/React.createElement("p", null, "GitHub Repository: ", /*#__PURE__*/React.createElement("a", {
    href: "https://github.com/Salmonyeonwoo/Portfolio-for-application-of-AI-Job-offers",
    target: "_blank",
    className: "text-cyan-400 hover:underline"
  }, "Portfolio-for-application-of-AI-Job-offers"))), contactModalOpen && /*#__PURE__*/React.createElement("div", {
    className: "fixed inset-0 bg-black/75 backdrop-blur-sm flex justify-center items-center z-50 p-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-900 border border-slate-700 text-slate-100 rounded-3xl p-6 sm:p-8 max-w-md w-full relative shadow-2xl space-y-5 animate-in fade-in"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setContactModalOpen(false),
    className: "absolute top-4 right-5 text-slate-400 hover:text-white text-2xl font-bold transition-colors"
  }, "\xD7"), /*#__PURE__*/React.createElement("h3", {
    className: "text-xl font-black text-white text-center border-b border-slate-800 pb-3"
  }, lang === 'jp' ? '連絡先情報 (Contact Info)' : lang === 'en' ? 'Contact Information' : '연락처 정보 (Contact Info)'), /*#__PURE__*/React.createElement("div", {
    className: "space-y-4 text-xs sm:text-sm"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "font-bold text-cyan-400 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-brands fa-line text-green-400"
  }), /*#__PURE__*/React.createElement("i", {
    className: "fa-brands fa-whatsapp text-emerald-400"
  }), /*#__PURE__*/React.createElement("span", null, lang === 'jp' ? 'メッセンジャー (Kakao / LINE / WhatsApp)' : lang === 'en' ? 'Instant Messengers (Kakao / LINE / WhatsApp)' : '메신저 (Kakao / LINE / WhatsApp)')), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-300 font-mono text-xs"
  }, "+6010-968-1392 (Malaysia)"), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-300 font-mono text-xs"
  }, "+8210-2153-1208 (Korea)")), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "font-bold text-rose-400 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-envelope"
  }), /*#__PURE__*/React.createElement("span", null, lang === 'jp' ? 'メールアドレス (Email)' : lang === 'en' ? 'Direct Email' : '이메일 (Email)')), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-300 font-mono text-xs"
  }, "yeonwoopark40@gmail.com"), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-300 font-mono text-xs"
  }, "yunwoopk@gmail.com"), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-300 font-mono text-xs"
  }, "yunwoopk@hanmail.net"))), /*#__PURE__*/React.createElement("button", {
    onClick: () => setContactModalOpen(false),
    className: "w-full bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-black py-2.5 rounded-xl transition-all text-xs"
  }, lang === 'jp' ? '閉じる (Close)' : lang === 'en' ? 'Close' : '닫기 (Close)'))), /*#__PURE__*/React.createElement("div", {
    className: "fixed bottom-6 right-6 z-50"
  }, /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => setIsChatOpen(prev => !prev),
    className: "group relative w-14 h-14 rounded-full bg-gradient-to-tr from-cyan-600 via-cyan-500 to-cyan-400 hover:from-cyan-500 hover:to-cyan-300 text-slate-950 flex items-center justify-center shadow-xl shadow-cyan-500/30 hover:scale-105 active:scale-95 transition-all duration-200 cursor-pointer",
    title: isChatOpen ? botI18n[lang]?.closeTooltip || "채팅창 닫기" : botI18n[lang]?.title || "David AI Simulator"
  }, isChatOpen ? /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-xmark text-2xl transition-transform group-hover:rotate-90"
  }) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-comments text-2xl"
  }), /*#__PURE__*/React.createElement("span", {
    className: "absolute -top-1 -right-1 flex h-4 w-4"
  }, /*#__PURE__*/React.createElement("span", {
    className: "animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-300 opacity-75"
  }), /*#__PURE__*/React.createElement("span", {
    className: "relative inline-flex rounded-full h-4 w-4 bg-cyan-400 border-2 border-slate-950"
  }))))), isChatOpen && /*#__PURE__*/React.createElement("div", {
    className: "fixed bottom-24 right-4 sm:right-6 z-50 w-[360px] sm:w-[420px] max-w-[calc(100vw-2rem)] h-[580px] max-h-[calc(100vh-8rem)] flex flex-col rounded-2xl bg-slate-900/95 border border-slate-700/80 shadow-2xl shadow-black/80 backdrop-blur-xl overflow-hidden animate-in fade-in slide-in-from-bottom-5 duration-200"
  }, /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-950/90 border-b border-slate-800 px-4 py-3 flex items-center justify-between"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "relative w-9 h-9 rounded-full bg-cyan-500/20 border border-cyan-400/40 flex items-center justify-center text-cyan-400"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-robot text-base"
  }), /*#__PURE__*/React.createElement("span", {
    className: "absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-slate-950"
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "text-sm font-bold text-white tracking-tight"
  }, botI18n[lang]?.title || "David AI Protocol Bot"), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-bold bg-cyan-950 text-cyan-300 border border-cyan-500/40 rounded px-1.5 py-0.5"
  }, botI18n[lang]?.badge || "SOP DEMO")), /*#__PURE__*/React.createElement("p", {
    className: "text-[11px] text-slate-400 flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"
  }), botI18n[lang]?.subStatus || "스샷 / PDF 증빙 분석 지원"))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1"
  }, /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => {
      setActiveCategory('auto');
      setAttachedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setMessages([{
        sender: 'ai',
        text: botI18n[lang]?.resetNotice || botI18n.kr.resetNotice
      }]);
    },
    className: "text-slate-400 hover:text-slate-200 p-1.5 rounded-lg hover:bg-slate-800/80 transition-colors text-xs",
    title: botI18n[lang]?.resetTooltip || "대화 초기화"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-rotate-right"
  })), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => setIsChatOpen(false),
    className: "text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800/80 transition-colors text-sm",
    title: botI18n[lang]?.closeTooltip || "닫기"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-xmark"
  })))), /*#__PURE__*/React.createElement("div", {
    className: "px-3 py-2 bg-slate-950/80 border-b border-slate-800/80 flex items-center gap-1.5 overflow-x-auto scrollbar-hide"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-slate-500 font-mono shrink-0"
  }, botI18n[lang]?.categoryLabel || "상품분류:"), [{
    id: 'auto',
    label: botI18n[lang]?.categories.auto || '⚡ 자동분석'
  }, {
    id: 'hotel',
    label: botI18n[lang]?.categories.hotel || '🏨 숙박/호텔'
  }, {
    id: 'flight',
    label: botI18n[lang]?.categories.flight || '✈️ 항공권'
  }, {
    id: 'commerce',
    label: botI18n[lang]?.categories.commerce || '📦 배송/커머스'
  }, {
    id: 'payout',
    label: botI18n[lang]?.categories.payout || '📊 재무/정산'
  }, {
    id: 'dispute',
    label: botI18n[lang]?.categories.dispute || '🛡️ 결제분쟁'
  }].map(cat => /*#__PURE__*/React.createElement("button", {
    key: cat.id,
    type: "button",
    onClick: () => setActiveCategory(cat.id),
    className: `px-2.5 py-1 rounded-full text-[11px] font-medium shrink-0 transition-all cursor-pointer ${activeCategory === cat.id ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm shadow-cyan-500/30' : 'bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/50'}`
  }, cat.label))), /*#__PURE__*/React.createElement("div", {
    className: "flex-1 overflow-y-auto p-4 space-y-3.5 scrollbar-hide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-center my-0.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] uppercase font-mono text-slate-500 bg-slate-950/60 border border-slate-800/80 px-2.5 py-0.5 rounded-full"
  }, botI18n[lang]?.decisionTreeTitle || "Multi-Category Decision Tree with Document Verification")), messages.map((msg, idx) => /*#__PURE__*/React.createElement("div", {
    key: idx,
    className: `flex items-start gap-2.5 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`
  }, msg.sender === 'ai' && /*#__PURE__*/React.createElement("div", {
    className: "w-7 h-7 rounded-full bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 text-xs shrink-0 mt-0.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-robot"
  })), /*#__PURE__*/React.createElement("div", {
    className: `rounded-2xl px-3.5 py-2.5 text-xs sm:text-sm leading-relaxed ${msg.sender === 'user' ? 'bg-gradient-to-r from-cyan-500 to-cyan-400 text-slate-950 font-medium rounded-tr-xs shadow-md shadow-cyan-500/10 max-w-[85%]' : 'bg-slate-800/90 border border-slate-700/80 text-slate-200 rounded-tl-xs shadow-md max-w-[88%]'}`
  }, msg.sender === 'ai' ? /*#__PURE__*/React.createElement("div", {
    className: "space-y-1.5"
  }, msg.text.split('\n').map((line, lIdx) => {
    const parts = line.split(/(\*\*.*?\*\*)/g);
    return /*#__PURE__*/React.createElement("div", {
      key: lIdx,
      className: "min-h-[1em]"
    }, parts.map((part, pIdx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return /*#__PURE__*/React.createElement("strong", {
          key: pIdx,
          className: "text-cyan-300 font-bold"
        }, part.slice(2, -2));
      }
      return part;
    }));
  })) : /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "whitespace-pre-line"
  }, msg.text), msg.attachment && (msg.attachment.type.startsWith('image/') && msg.attachment.previewUrl ? /*#__PURE__*/React.createElement("div", {
    className: "mt-2.5 rounded-xl overflow-hidden border border-slate-950/40 bg-slate-950/80 max-w-[240px]"
  }, /*#__PURE__*/React.createElement("img", {
    src: msg.attachment.previewUrl,
    alt: msg.attachment.name,
    className: "w-full max-h-36 object-cover cursor-pointer hover:opacity-95 transition-opacity",
    onClick: () => window.open(msg.attachment.previewUrl, '_blank'),
    title: lang === "jp" ? "クリックして原本を表示" : lang === "en" ? "Click to view original" : "클릭하여 원본 보기"
  }), /*#__PURE__*/React.createElement("div", {
    className: "p-1.5 flex items-center justify-between text-[10px] font-mono text-slate-300 border-t border-slate-800/60"
  }, /*#__PURE__*/React.createElement("span", {
    className: "truncate max-w-[140px]"
  }, "\uD83D\uDCCE ", msg.attachment.name), /*#__PURE__*/React.createElement("span", null, (msg.attachment.size / 1024).toFixed(1), " KB"))) : /*#__PURE__*/React.createElement("div", {
    className: "mt-2.5 flex items-center gap-2.5 p-2.5 rounded-xl bg-slate-950/90 border border-slate-800 text-slate-200 max-w-[260px]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-8 h-8 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-400 flex items-center justify-center text-sm font-bold shrink-0"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-file-pdf"
  })), /*#__PURE__*/React.createElement("div", {
    className: "truncate flex-1"
  }, /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-white font-medium truncate"
  }, msg.attachment.name), /*#__PURE__*/React.createElement("p", {
    className: "text-[10px] text-slate-400 font-mono"
  }, (msg.attachment.size / 1024).toFixed(1), " KB \u2022 ", msg.attachment.type.includes('pdf') ? botI18n[lang]?.pdfDoc || 'PDF 문서' : lang === 'jp' ? 'ファイル' : lang === 'en' ? 'File' : '파일')), /*#__PURE__*/React.createElement("span", {
    className: "text-xs text-emerald-400"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-circle-check"
  })))))))), isTyping && /*#__PURE__*/React.createElement("div", {
    className: "flex items-start gap-2.5 justify-start"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-7 h-7 rounded-full bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 text-xs shrink-0 mt-0.5"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-robot"
  })), /*#__PURE__*/React.createElement("div", {
    className: "bg-slate-800/90 border border-slate-700/80 text-slate-400 rounded-2xl rounded-tl-xs px-4 py-3 shadow-md flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-2 h-2 rounded-full bg-cyan-400 animate-bounce"
  }), /*#__PURE__*/React.createElement("span", {
    className: "w-2 h-2 rounded-full bg-cyan-400 animate-bounce [animation-delay:0.2s]"
  }), /*#__PURE__*/React.createElement("span", {
    className: "w-2 h-2 rounded-full bg-cyan-400 animate-bounce [animation-delay:0.4s]"
  }), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] text-cyan-300 ml-1.5 font-mono"
  }, botI18n[lang]?.typing || "증빙 자료 및 약관 대조 분석 중..."))), /*#__PURE__*/React.createElement("div", {
    ref: chatEndRef
  })), /*#__PURE__*/React.createElement("div", {
    className: "px-3 py-2 flex flex-wrap gap-1.5 bg-slate-950/90 border-t border-slate-800/80"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-slate-500 w-full flex items-center gap-1 font-mono"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-wand-magic-sparkles text-cyan-400"
  }), " ", botI18n[lang]?.scenarioLabel || "검증 시나리오 및 증빙 테스트:"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => handleSendMessage(lang === 'jp' ? "ポリシーを確認せずに、すぐキャンセルできるのでしょうか？" : lang === 'en' ? "Can I cancel immediately without checking any policies?" : "바로 취소가 되는 건가요? 정책을 안 보고요?"),
    disabled: isTyping,
    className: "text-[11px] bg-slate-900 hover:bg-slate-800 text-amber-300 border border-amber-500/40 rounded-full px-2.5 py-1 transition-all hover:border-amber-400 disabled:opacity-40 cursor-pointer flex items-center gap-1 font-medium"
  }, /*#__PURE__*/React.createElement("span", null, "\u26A0\uFE0F"), " ", botI18n[lang]?.scenarios.blind || "정책 미확인 즉시 취소 문의"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => handleSendMessage(lang === 'jp' ? "欠航証明書PDFを添付し、取消手数料免除（Waiver）を申請します。" : lang === 'en' ? "Submitting flight cancellation notice PDF for penalty waiver review." : "결항 확인서 PDF 첨부하여 취소 수수료 면제(Waiver) 요청합니다.", {
      name: lang === 'jp' ? "フライト欠航証明書_証憑.pdf" : lang === 'en' ? "Flight_Cancellation_Notice.pdf" : "항공편_결항확인서_증빙.pdf",
      size: 245760,
      type: "application/pdf"
    }),
    disabled: isTyping,
    className: "text-[11px] bg-slate-900 hover:bg-slate-800 text-rose-300 border border-rose-500/40 rounded-full px-2.5 py-1 transition-all hover:border-rose-400 disabled:opacity-40 cursor-pointer flex items-center gap-1 font-medium"
  }, /*#__PURE__*/React.createElement("span", null, "\uD83D\uDCCE"), " ", botI18n[lang]?.scenarios.flightWaiver || "결항 확인서 PDF 첨부"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => handleSendMessage(lang === 'jp' ? "SAP運賃インボイスのスクショ証憑を提出いたします。" : lang === 'en' ? "Submitting SAP freight invoice screenshot for ledger variance audit." : "SAP 운임 인보이스 스크린샷 증빙 제출합니다.", {
      name: lang === 'jp' ? "SAP_運賃請求書_差額証憑.png" : lang === 'en' ? "SAP_Freight_Invoice_Variance.png" : "SAP_인보이스_운임차액_증빙.png",
      size: 412500,
      type: "image/png",
      previewUrl: "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%230f172a'><rect width='100' height='100' rx='8'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='%2338bdf8' font-size='11' font-family='monospace'>SAP_INV.PNG</text></svg>"
    }),
    disabled: isTyping,
    className: "text-[11px] bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-emerald-500/40 rounded-full px-2.5 py-1 transition-all hover:border-emerald-400 disabled:opacity-40 cursor-pointer flex items-center gap-1 font-medium"
  }, /*#__PURE__*/React.createElement("span", null, "\uD83D\uDCCE"), " ", botI18n[lang]?.scenarios.invoice || "인보이스 스크린샷 첨부"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => handleSendMessage(lang === 'jp' ? "返金不可特約プランのホテル宿泊ですが、取消可能でしょうか？" : lang === 'en' ? "Is cancellation possible for a non-refundable promotional hotel rate?" : "호텔 환불불가 특가 상품인데 취소 가능한가요?"),
    disabled: isTyping,
    className: "text-[11px] bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-cyan-500/30 rounded-full px-2.5 py-1 transition-all hover:border-cyan-400 disabled:opacity-40 cursor-pointer flex items-center gap-1"
  }, /*#__PURE__*/React.createElement("span", null, "\uD83C\uDFE8"), " ", botI18n[lang]?.scenarios.hotelSpecial || "호텔 환불불가 특가"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => handleSendMessage(lang === 'jp' ? "すでに出荷完了済みの商品について返品・返金を希望します。" : lang === 'en' ? "Requesting return and refund for an order that has already been dispatched." : "출고 완료된 상품 반품 및 환불 요청합니다."),
    disabled: isTyping,
    className: "text-[11px] bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-cyan-500/30 rounded-full px-2.5 py-1 transition-all hover:border-cyan-400 disabled:opacity-40 cursor-pointer flex items-center gap-1"
  }, /*#__PURE__*/React.createElement("span", null, "\uD83D\uDCE6"), " ", botI18n[lang]?.scenarios.commerceTransit || "출고 후 반품 배송비")), attachedFile && /*#__PURE__*/React.createElement("div", {
    className: "px-3 py-2 bg-slate-950/95 border-t border-slate-800/90 flex items-center justify-between animate-in fade-in duration-150"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2.5 max-w-[85%]"
  }, attachedFile.type.startsWith('image/') && attachedFile.previewUrl ? /*#__PURE__*/React.createElement("img", {
    src: attachedFile.previewUrl,
    alt: "preview",
    className: "w-9 h-9 rounded-lg object-cover border border-cyan-500/40 shadow-sm"
  }) : /*#__PURE__*/React.createElement("div", {
    className: "w-9 h-9 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-400 flex items-center justify-center text-sm font-bold shrink-0"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-file-pdf"
  })), /*#__PURE__*/React.createElement("div", {
    className: "truncate"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950 border border-cyan-500/30 rounded px-1"
  }, attachedFile.type.includes('pdf') ? botI18n[lang]?.pdfDoc || 'PDF 증빙' : botI18n[lang]?.screenshotProof || '스크린샷 증빙'), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-200 font-medium truncate"
  }, attachedFile.name)), /*#__PURE__*/React.createElement("p", {
    className: "text-[10px] text-slate-400 font-mono"
  }, (attachedFile.size / 1024).toFixed(1), " KB \u2022 ", botI18n[lang]?.readyToSend || "전송 준비 완료"))), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => {
      setAttachedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    },
    className: "text-slate-400 hover:text-rose-400 p-1.5 rounded-lg hover:bg-slate-800 transition-colors text-sm",
    title: botI18n[lang]?.cancelAttachTooltip || "첨부 취소"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-xmark"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "p-3 bg-slate-950 border-t border-slate-800 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("input", {
    type: "file",
    ref: fileInputRef,
    onChange: handleFileSelect,
    accept: "image/*,.pdf",
    className: "hidden"
  }), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => fileInputRef.current?.click(),
    disabled: isTyping,
    className: "w-10 h-10 text-slate-400 hover:text-cyan-400 hover:bg-slate-800/80 active:scale-95 disabled:opacity-40 rounded-xl transition-all flex items-center justify-center cursor-pointer shrink-0",
    title: botI18n[lang]?.attachTooltip || "증빙 자료 첨부 (스크린샷, 이미지, PDF)"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-paperclip text-sm"
  })), /*#__PURE__*/React.createElement("input", {
    type: "text",
    value: inputValue,
    onChange: e => setInputValue(e.target.value),
    onKeyDown: handleKeyDown,
    placeholder: attachedFile ? botI18n[lang]?.placeholderAttached || "증빙 서류 관련 설명 입력 (선택사항)..." : botI18n[lang]?.placeholderDefault || "문의 입력 또는 📎 서류 첨부...",
    disabled: isTyping,
    className: "flex-1 bg-slate-900 border border-slate-700 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 text-slate-100 placeholder-slate-500 text-xs sm:text-sm rounded-xl px-3.5 py-2.5 outline-none transition-colors disabled:opacity-60"
  }), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => handleSendMessage(),
    disabled: isTyping || !inputValue.trim() && !attachedFile,
    className: "w-10 h-10 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:hover:bg-cyan-500 text-slate-950 font-bold rounded-xl transition-all flex items-center justify-center cursor-pointer shadow-md shadow-cyan-500/20 shrink-0",
    title: botI18n[lang]?.sendTooltip || "전송"
  }, /*#__PURE__*/React.createElement("i", {
    className: "fa-solid fa-paper-plane text-sm"
  })))));
}
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render( /*#__PURE__*/React.createElement(App, null));