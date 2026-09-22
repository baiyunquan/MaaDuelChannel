from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class InstructionGuideDialog(QDialog):
    """Independent operation instructions and shortcuts guide window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MaaDuelChannel 审核客户端操作指引与快捷键说明")
        self.resize(720, 640)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QLabel("MaaDuelChannel 本地审核与标注操作指引")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)

        # Content browser (rich text)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            """
            QTextBrowser {
                background-color: #1e1e1e;
                color: #dddddd;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 10px;
                font-family: sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
            """
        )

        content = """
        <style>
            h2 {
                color: #00d4ff; font-size: 14px; margin-top: 10px; margin-bottom: 4px;
                border-bottom: 1px solid #333; padding-bottom: 2px;
            }
            h3 { color: #ffaa33; font-size: 13px; margin-top: 8px; margin-bottom: 2px; }
            p, li { color: #cccccc; font-size: 12px; }
            code {
                background-color: #2a2a2a; color: #00ffaa;
                padding: 2px 5px; border-radius: 3px; font-family: monospace;
            }
            table { border-collapse: collapse; width: 100%; margin-top: 6px; margin-bottom: 10px; }
            th {
                background-color: #2d2d2d; color: #ffffff; border: 1px solid #444;
                padding: 5px 8px; text-align: left;
            }
            td { border: 1px solid #3c3c3c; padding: 5px 8px; color: #cccccc; }
            tr:nth-child(even) { background-color: #242424; }
            .step-box {
                background-color: #25282c; border-left: 4px solid #007acc;
                padding: 6px 10px; margin-bottom: 8px; border-radius: 3px;
            }
            .tip-box {
                background-color: #2d271e; border-left: 4px solid #ffaa00;
                padding: 6px 10px; margin-bottom: 8px; border-radius: 3px;
            }
        </style>

        <div class="tip-box">
            <b>核心原则：</b>本客户端用于构建高质量 Hard Set 数据集。
            审核标准为：卡槽阵容敌人数量必须与战场实际拉框完全一致，且选定胜负方。
        </div>

        <h2>三步标准审核流程</h2>

        <div class="step-box">
            <h3>Step 1: 审核三个证据帧时间位置 (左下控制区)</h3>
            <ul>
                <li><b>切换证据帧：</b>点击 <code>1. 准备帧 (Prep)</code> /
                    <code>2. 站位帧 (Layout)</code> / <code>3. 结算帧 (End)</code>，
                    时间滑块将自动跳转至对应时间点；</li>
                <li><b>帧微调：</b>拖动时间滑块，或点击 <code>[-1帧]</code> / <code>[+1帧]</code> /
                    <code>[-0.1s]</code> / <code>[+0.1s]</code> 按钮微调，实时查看画面；</li>
                <li><b>重新落盘：</b>定位满意后，点击 <code>[设为当前证据帧并重新落盘]</code>，
                    系统会自动覆盖存储帧图片并更新时间戳。若修改的是 Layout 帧，中央画布会即时更新底图。</li>
            </ul>
        </div>

        <div class="step-box">
            <h3>Step 2: 审核 6 个卡槽的敌人头像与数量 (右侧阵容区)</h3>
            <ul>
                <li><b>卡槽结构：</b>上方为左方 3 槽（Slot 1~3），下方为右方 3 槽（Slot 4~6）；</li>
                <li><b>选择与替换敌人：</b>
                    <ol>
                        <li>点击任意卡槽将其激活（出现高亮亮框）；</li>
                        <li>在右下方 <b>敌人图谱</b> 中通过搜索栏（支持常用名/原名/ID）找到目标敌人；</li>
                        <li>点击敌人微缩头像，即可将该敌人赋予当前激活卡槽；</li>
                    </ol>
                </li>
                <li><b>调整数量与清空：</b>通过每个卡槽右侧的数字输入框调整数量；若某槽位未上场，
                    可点击 <code>X</code> 按钮清空该槽；</li>
                <li><b>配额追踪状态：</b>卡槽右下角实时显示配额对比：
                    <ul>
                        <li><code>[OK] 3/3</code>（绿色）：画布拉框数量与卡槽数量已完全吻合；</li>
                        <li><code>[待补] 2/3</code>（黄色）：画布拉框数量少于卡槽设定，需补拉框；</li>
                        <li><code>! 4/3</code>（红色）：画布拉框数量已超出卡槽设定，需删除多余框。</li>
                    </ul>
                </li>
            </ul>
        </div>

        <div class="step-box">
            <h3>Step 3: 战场交互拉框与快捷键赋权 (中央画布区)</h3>
            <ul>
                <li><b>画布漫游：</b>鼠标滚轮缩放；按住 <code>鼠标中键拖拽</code> 或
                    <code>Shift + 左键拖拽</code> 平移画面；</li>
                <li><b>切换模式：</b>
                    <ul>
                        <li>按 <code>V</code> 键：切换为 <b>选择/移动</b> 模式。
                            点击框选中，选中后拖动框内部移动，拖动四周 8 个手柄缩放尺寸；</li>
                        <li>按 <code>R</code> 键：切换为 <b>绘制新框</b> 模式。
                            按住鼠标左键拖拽即可创建新框（自动继承当前激活卡槽的阵营与类别）；</li>
                    </ul>
                </li>
                <li><b>快捷键赋权：</b>选中某个标注框后，<b>直接按下数字键 <code>1</code> ~ <code>6</code></b>，
                    该框将瞬间变更为对应卡槽的阵营与敌人类型！</li>
                <li><b>删除框：</b>选中不需要的框，按键盘 <code>Delete</code> 或
                    <code>Backspace</code> 键即可删除。</li>
                <li><b>地面接触落点：</b>每个框底边中心带有黄色圆点，代表单位在战场地面接触站位坐标点。</li>
            </ul>
        </div>

        <h2>快捷键速查表</h2>
        <table>
            <tr>
                <th width="28%">快捷键</th>
                <th width="32%">功能操作</th>
                <th width="40%">说明</th>
            </tr>
            <tr>
                <td><code>A</code></td>
                <td>上一局样本</td>
                <td>保存当前草稿并切换至前一局</td>
            </tr>
            <tr>
                <td><code>D</code></td>
                <td>下一局样本</td>
                <td>保存当前草稿并切换至下一局</td>
            </tr>
            <tr>
                <td><code>V</code></td>
                <td>选择 / 调整模式</td>
                <td>用于选中框、拖动移动、拉伸 8 手柄</td>
            </tr>
            <tr>
                <td><code>R</code></td>
                <td>绘制新框模式</td>
                <td>按住鼠标左键在画布上拉矩形框</td>
            </tr>
            <tr>
                <td><code>1</code> ~ <code>6</code></td>
                <td>绑定选中框 / 激活卡槽</td>
                <td>有选中框时直接赋予该卡槽类别；无选框时切换卡槽</td>
            </tr>
            <tr>
                <td><code>Delete</code> / <code>Backspace</code></td>
                <td>删除选中框</td>
                <td>删除当前激活的战场标注框</td>
            </tr>
            <tr>
                <td><code>鼠标滚轮</code></td>
                <td>画布缩放</td>
                <td>以当前鼠标光标所在像素为锚点无级缩放</td>
            </tr>
            <tr>
                <td><code>鼠标中键拖拽</code></td>
                <td>画布平移</td>
                <td>或按住 <code>Shift + 鼠标左键拖拽</code> 漫游画布</td>
            </tr>
            <tr>
                <td><code>F1</code></td>
                <td>打开/关闭操作指引</td>
                <td>随时唤出此独立帮助指南窗口</td>
            </tr>
        </table>

        <h2>审核通过与保存规则</h2>
        <ul>
            <li>选定胜方：勾选 <b>左方胜</b> 或 <b>右方胜</b>；</li>
            <li>点击 <b>[通过 (Accept)]</b> 前，必须确保左右两侧卡槽的所有配额均显示为
                <code>[OK]</code>（即每个种类的卡槽数量与战场上对应拉框总数完全相等）；若有不符，系统会弹窗提示具体差异；</li>
            <li>点击 <b>[驳回 (Reject)]</b> 将样本标记为驳回状态，便于后续过滤排除。</li>
        </ul>
        """

        browser.setHtml(content)
        layout.addWidget(browser)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("我知道了 (关闭)")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #007acc; color: #ffffff; padding: 6px 16px; "
            "border-radius: 4px; font-weight: bold; border: none; }\n"
            "QPushButton:hover { background-color: #0099ff; }"
        )
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
