param(
    [string]$OutputPath = (Join-Path $PSScriptRoot 'output\SSU_Data_Dash_5min_Presentation_v2.pptx')
)

$ErrorActionPreference = 'Stop'

function Rgb([string]$hex) {
    $clean = $hex.TrimStart('#')
    $r = [Convert]::ToInt32($clean.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($clean.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($clean.Substring(4, 2), 16)
    return $r + (256 * $g) + (65536 * $b)
}

$C = @{
    Navy = Rgb '#0B2239'
    DeepNavy = Rgb '#071827'
    Teal = Rgb '#1E7F7A'
    LightTeal = Rgb '#D9ECE8'
    Coral = Rgb '#F26B4A'
    LightCoral = Rgb '#FBE1D8'
    Cream = Rgb '#F6F1E8'
    White = Rgb '#FFFFFF'
    Ink = Rgb '#17232E'
    Gray = Rgb '#667786'
    LightGray = Rgb '#D8E0E5'
    Pale = Rgb '#EEF3F4'
}

$ppLayoutBlank = 12
$ppSaveAsOpenXMLPresentation = 24
$msoFalse = 0
$msoTrue = -1
$msoTextOrientationHorizontal = 1
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoShapeOval = 9
$ppAlignLeft = 1
$ppAlignCenter = 2
$ppAlignRight = 3
$ppPlaceholderBody = 2
$xlBarClustered = 57
$xlColumnClustered = 51
$xlCategory = 1
$xlValue = 2
$xlPrimary = 1
$xlLegendPositionBottom = -4107
$xlLabelPositionOutsideEnd = 2

function Add-Text {
    param($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [double]$H,
          [double]$Size = 20, [int]$Color = $C.Ink, [bool]$Bold = $false,
          [int]$Align = $ppAlignLeft, [string]$Font = 'Aptos', [double]$Margin = 0)
    $shape = $Slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $X, $Y, $W, $H)
    $shape.Fill.Visible = $msoFalse
    $shape.Line.Visible = $msoFalse
    $shape.TextFrame.MarginLeft = $Margin
    $shape.TextFrame.MarginRight = $Margin
    $shape.TextFrame.MarginTop = $Margin
    $shape.TextFrame.MarginBottom = $Margin
    $shape.TextFrame.WordWrap = $msoTrue
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = $Font
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Bold = if ($Bold) { $msoTrue } else { $msoFalse }
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Box {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H,
          [int]$Fill, [int]$Line = $Fill, [double]$Radius = 0)
    $kind = if ($Radius -gt 0) { $msoShapeRoundedRectangle } else { $msoShapeRectangle }
    $shape = $Slide.Shapes.AddShape($kind, $X, $Y, $W, $H)
    $shape.Fill.ForeColor.RGB = $Fill
    $shape.Fill.Solid()
    $shape.Line.ForeColor.RGB = $Line
    $shape.Line.Weight = 1
    return $shape
}

function Add-Title {
    param($Slide, [string]$Title, [int]$Number)
    Add-Text $Slide $Title 54 28 800 48 29 $C.Navy $true | Out-Null
    Add-Text $Slide ([string]$Number) 880 33 28 25 12 $C.Gray $false $ppAlignRight | Out-Null
    $line = $Slide.Shapes.AddLine(54, 78, 906, 78)
    $line.Line.ForeColor.RGB = $C.LightGray
    $line.Line.Weight = 1
}

function Add-Notes {
    param($Slide, [string]$Text)
    $notes = $Slide.NotesPage
    foreach ($shape in $notes.Shapes) {
        try {
            if ($shape.Type -eq 14 -and $shape.PlaceholderFormat.Type -eq $ppPlaceholderBody) {
                $shape.TextFrame.TextRange.Text = $Text
                return
            }
        } catch { }
    }
}

function Add-TableText {
    param($Cell, [string]$Text, [double]$Size, [int]$Color, [bool]$Bold, [int]$Align)
    $Cell.Shape.TextFrame.TextRange.Text = $Text
    $Cell.Shape.TextFrame.TextRange.Font.Name = 'Aptos'
    $Cell.Shape.TextFrame.TextRange.Font.Size = $Size
    $Cell.Shape.TextFrame.TextRange.Font.Bold = if ($Bold) { $msoTrue } else { $msoFalse }
    $Cell.Shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $Cell.Shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    $Cell.Shape.TextFrame.VerticalAnchor = 3
    $Cell.Shape.TextFrame.MarginLeft = 6
    $Cell.Shape.TextFrame.MarginRight = 6
}

function Add-NativeChart {
    param($Slide, [int]$ChartType, [double]$X, [double]$Y, [double]$W, [double]$H,
          [string[]]$Categories, [object[]]$Series, [double]$Maximum,
          [string]$NumberFormat = '0.00%')

    $workbook = $script:excel.Workbooks.Add()
    $sheet = $workbook.Worksheets.Item(1)
    $sheet.Cells.Clear()
    $sheet.Cells.Item(1, 1).Value2 = ''
    for ($s = 0; $s -lt $Series.Count; $s++) {
        $sheet.Cells.Item(1, $s + 2).Value2 = $Series[$s].Name
    }
    for ($i = 0; $i -lt $Categories.Count; $i++) {
        $sheet.Cells.Item($i + 2, 1).Value2 = $Categories[$i]
        for ($s = 0; $s -lt $Series.Count; $s++) {
            $sheet.Cells.Item($i + 2, $s + 2).Value2 = [double]$Series[$s].Values[$i]
        }
    }
    $lastCell = $sheet.Cells.Item($Categories.Count + 1, $Series.Count + 1)
    $range = $sheet.Range($sheet.Cells.Item(1, 1), $lastCell)
    $chartObject = $sheet.ChartObjects().Add(0, 0, $W, $H)
    $chart = $chartObject.Chart
    $chart.ChartType = $ChartType
    $chart.SetSourceData($range)
    $chart.PlotBy = 2
    $chart.HasTitle = $false
    $chart.HasLegend = $true
    $chart.Legend.Position = $xlLegendPositionBottom
    $chart.Legend.Font.Name = 'Aptos'
    $chart.Legend.Font.Size = 11
    $chart.ChartArea.Format.Fill.Visible = $msoFalse
    $chart.ChartArea.Format.Line.Visible = $msoFalse
    $chart.PlotArea.Format.Fill.Visible = $msoFalse
    $chart.PlotArea.Format.Line.Visible = $msoFalse
    $chart.Axes($xlValue, $xlPrimary).MinimumScale = 0
    $chart.Axes($xlValue, $xlPrimary).MaximumScale = $Maximum
    $chart.Axes($xlValue, $xlPrimary).TickLabels.NumberFormat = '0%'
    $chart.Axes($xlValue, $xlPrimary).TickLabels.Font.Name = 'Aptos'
    $chart.Axes($xlValue, $xlPrimary).TickLabels.Font.Size = 10
    $chart.Axes($xlCategory, $xlPrimary).TickLabels.Font.Name = 'Aptos'
    $chart.Axes($xlCategory, $xlPrimary).TickLabels.Font.Size = 11
    $chart.Axes($xlValue, $xlPrimary).MajorGridlines.Format.Line.ForeColor.RGB = $C.LightGray
    for ($s = 1; $s -le $Series.Count; $s++) {
        $seriesItem = $chart.SeriesCollection($s)
        $seriesItem.Format.Fill.ForeColor.RGB = $Series[$s - 1].Color
        $seriesItem.Format.Line.Visible = $msoFalse
        $seriesItem.ApplyDataLabels()
        $seriesItem.DataLabels().Position = $xlLabelPositionOutsideEnd
        $seriesItem.DataLabels().NumberFormat = $NumberFormat
        $seriesItem.DataLabels().Font.Name = 'Aptos'
        $seriesItem.DataLabels().Font.Size = 10
        $seriesItem.DataLabels().Font.Color = $C.Ink
    }
    $chartObject.Copy()
    [System.Threading.Thread]::Sleep(300)
    $pasted = $Slide.Shapes.Paste()
    $chartShape = $pasted.Item(1)
    $chartShape.Left = $X
    $chartShape.Top = $Y
    $chartShape.Width = $W
    $chartShape.Height = $H
    $workbook.Close($false)
    return $chartShape
}

$coverImage = Join-Path $PSScriptRoot 'assets\toronto_home_valuation_cover.png'
$previewDir = Join-Path $PSScriptRoot '.build\rendered'
New-Item -ItemType Directory -Force -Path (Split-Path $OutputPath), $previewDir | Out-Null

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = $msoTrue
$script:excel = New-Object -ComObject Excel.Application
$script:excel.Visible = $false
$script:excel.DisplayAlerts = $false
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 540

try {
    # Slide 1
    $s = $presentation.Slides.Add(1, $ppLayoutBlank)
    $s.Shapes.AddPicture($coverImage, $msoFalse, $msoTrue, 0, 0, 960, 540) | Out-Null
    $overlay = Add-Box $s 0 0 520 540 $C.DeepNavy $C.DeepNavy
    $overlay.Fill.Transparency = 0.08
    Add-Text $s "Predicting Toronto`nhome sale prices" 58 112 450 138 35 $C.White $true | Out-Null
    Add-Text $s 'A leakage-safe ensemble for off-market and list-aware valuation' 62 274 390 68 18 $C.Cream $false | Out-Null
    Add-Text $s 'SSU Data Dash 2026' 62 372 330 30 14 $C.LightTeal $true | Out-Null
    Add-Text $s 'Preston and Jiang' 62 410 330 28 15 $C.White $false | Out-Null
    Add-Notes $s @'
We built a home price estimator for two situations. In the off-market track, we value a home without seeing the asking price. In the list-aware track, the asking price is available. Our goal was to improve accuracy without using any information that would not have existed on the prediction date.

Source: FINAL/handout.md. Cover illustration: OpenAI ImageGen, decorative only.
'@

    # Slide 2
    $s = $presentation.Slides.Add(2, $ppLayoutBlank)
    $s.Background.Fill.ForeColor.RGB = $C.Cream
    Add-Title $s 'Two valuation questions' 2
    Add-Text $s 'Before listing' 60 108 360 32 15 $C.Teal $true | Out-Null
    Add-Text $s 'OFF-MARKET' 60 145 360 44 28 $C.Navy $true | Out-Null
    Add-Text $s 'What is this home worth?' 60 198 360 32 21 $C.Ink $true | Out-Null
    Add-Text $s "Uses property details, location, date,`nand listing remarks`n`nNever uses list price" 60 252 350 126 17 $C.Gray $false | Out-Null
    Add-Text $s 'Counts for 20 accuracy points' 60 430 330 25 14 $C.Coral $true | Out-Null
    $divider = $s.Shapes.AddLine(480, 112, 480, 465)
    $divider.Line.ForeColor.RGB = $C.LightGray
    $divider.Line.Weight = 2
    Add-Text $s 'After listing' 540 108 360 32 15 $C.Coral $true | Out-Null
    Add-Text $s 'LIST-AWARE' 540 145 360 44 28 $C.Navy $true | Out-Null
    Add-Text $s 'What will this listing close at?' 540 198 360 58 21 $C.Ink $true | Out-Null
    Add-Text $s "Uses the same information`nplus the seller's list price`n`nA different and easier prediction task" 540 270 350 126 17 $C.Gray $false | Out-Null
    Add-Text $s 'Counts for 10 accuracy points' 540 430 330 25 14 $C.Coral $true | Out-Null
    Add-Notes $s @'
These tracks answer different questions. Off-market valuation is useful before a home is listed, so the model can use the property, location, date, and remarks, but never list price. List-aware valuation estimates the closing price after listing and can use the seller's asking price. That makes it easier, but it also means the two validation scores cannot be treated as the same task.

Source: FINAL/handout.md.
'@

    # Slide 3
    $s = $presentation.Slides.Add(3, $ppLayoutBlank)
    $s.Background.Fill.ForeColor.RGB = $C.White
    Add-Title $s 'The final ensemble' 3
    Add-Text $s 'The strongest signal came from earlier local sales' 54 96 620 35 21 $C.Teal $true | Out-Null
    $signals = @(
        @('HOME', 'size, lot, rooms, type'),
        @('LOCATION', 'community and map cells'),
        @('HISTORY', 'local price and price per sq. ft.'),
        @('TIME + TEXT', 'recency, season, listing remarks')
    )
    for ($i = 0; $i -lt $signals.Count; $i++) {
        $y = 148 + 63 * $i
        Add-Text $s $signals[$i][0] 60 $y 112 26 13 $C.Coral $true | Out-Null
        Add-Text $s $signals[$i][1] 174 $y 286 32 16 $C.Ink $false | Out-Null
    }
    $line = $s.Shapes.AddLine(482, 144, 482, 404)
    $line.Line.ForeColor.RGB = $C.LightGray
    $line.Line.Weight = 2
    Add-Text $s 'OFF-MARKET BLEND' 530 112 350 24 13 $C.Teal $true | Out-Null
    Add-Text $s '49%  XGBoost' 530 153 300 30 20 $C.Navy $true | Out-Null
    Add-Text $s '32%  recency-aware LightGBM' 530 194 350 30 18 $C.Ink $false | Out-Null
    Add-Text $s '19%  TF-IDF Ridge text model' 530 233 350 30 18 $C.Ink $false | Out-Null
    Add-Text $s 'LIST-AWARE BLEND' 530 294 350 24 13 $C.Coral $true | Out-Null
    Add-Text $s '56%  historical sold-to-list ratio' 530 331 360 30 18 $C.Navy $true | Out-Null
    Add-Text $s '25% recent LightGBM, 13% off-market anchor, 6% target-encoded LightGBM' 530 371 360 62 16 $C.Ink $false | Out-Null
    Add-Text $s 'Geometric blending combines models that make different percentage errors' 60 466 820 30 16 $C.Gray $false | Out-Null
    Add-Notes $s @'
The largest improvement came from teaching the model about earlier local sales. We created historical price and price-per-square-foot summaries for communities, property types, styles, and geographic cells. A second version gave more weight to recent transactions so the model could react to market drift. We combined those signals with physical home features, location, calendar variables, and listing language. Off-market uses XGBoost, a recency-aware LightGBM model, and a TF-IDF Ridge text model. List-aware relies mainly on the historical community sold-to-list ratio, supported by two LightGBM components and an independent off-market value estimate. Each track uses a geometric blend because the models predict in log-price space and make different errors.

Sources: gbt_ensemble_experiments/FINAL_MODEL.md and report/report.tex.
'@

    # Slide 4
    $s = $presentation.Slides.Add(4, $ppLayoutBlank)
    $s.Background.Fill.ForeColor.RGB = $C.Cream
    Add-Title $s 'Model selection through experiments' 4
    $tableShape = $s.Shapes.AddTable(4, 4, 54, 118, 852, 218)
    $table = $tableShape.Table
    $widths = @(240, 140, 140, 332)
    for ($col = 1; $col -le 4; $col++) { $table.Columns.Item($col).Width = $widths[$col - 1] }
    $headers = @('MODEL', 'OFF-MARKET', 'LIST-AWARE', 'DECISION')
    for ($col = 1; $col -le 4; $col++) {
        $cell = $table.Cell(1, $col)
        $cell.Shape.Fill.ForeColor.RGB = $C.Navy
        Add-TableText $cell $headers[$col - 1] 12 $C.White $true $ppAlignCenter
    }
    $rows = @(
        @('Final GBT ensemble', '7.27%', '3.98%', 'Selected'),
        @("Preston's PyTorch MLP", '11.19%', '5.63%', 'Useful learning, weaker result'),
        @("Jiang's CatBoost correction", '7.67%', '3.91%', 'Slight list-aware gain, too costly')
    )
    for ($row = 2; $row -le 4; $row++) {
        $fill = if ($row -eq 2) { $C.LightTeal } elseif ($row -eq 4) { $C.LightCoral } else { $C.White }
        for ($col = 1; $col -le 4; $col++) {
            $cell = $table.Cell($row, $col)
            $cell.Shape.Fill.ForeColor.RGB = $fill
            $align = if ($col -eq 1 -or $col -eq 4) { $ppAlignLeft } else { $ppAlignCenter }
            Add-TableText $cell $rows[$row - 2][$col - 1] 14 $C.Ink ($row -eq 2) $align
        }
    }
    Add-Text $s 'Why the MLP trailed' 56 370 250 28 16 $C.Teal $true | Out-Null
    Add-Text $s 'Its inputs omitted listing remarks and the richer historical local-price signals that helped the trees.' 56 404 370 58 16 $C.Ink $false | Out-Null
    Add-Text $s 'Why we rejected the small CatBoost gain' 512 370 360 28 16 $C.Coral $true | Out-Null
    Add-Text $s 'A cold six-month validation required 114 CatBoost fits, while off-market accuracy also fell.' 512 404 370 58 16 $C.Ink $false | Out-Null
    Add-Notes $s @'
We did not choose the most complicated model. Preston tested a dense PyTorch MLP with standardized numeric inputs and learned location embeddings. It reached 11.19 percent off-market and 5.63 percent list-aware MdAPE. Its feature set did not include remarks or the richer historical local-price signals that helped the tree ensemble. Jiang developed a CatBoost ensemble with a residual text correction. It achieved a slightly better list-aware score of 3.91 percent, but off-market was 7.67 percent and a cold six-month validation required 114 CatBoost fits. We selected the final GBT ensemble because it had the best off-market result, nearly matched the list-aware result, and was much more practical to reproduce.

Sources: mlp_expriments/reports/validation_report.json; jiang-attempts branch, checkpoints/text_corrected_ensemble_v11(ultra slow)/validation_report.json; jiang-attempts branch, EXPERIMENT_V12.md; gbt_ensemble_experiments/results/final_walk_forward_report.json.
'@

    # Slide 5
    $s = $presentation.Slides.Add(5, $ppLayoutBlank)
    $s.Background.Fill.ForeColor.RGB = $C.White
    Add-Title $s 'Walk-forward validation' 5
    Add-Text $s 'Blend selection' 64 114 220 24 14 $C.Teal $true | Out-Null
    Add-Text $s 'Reported validation' 322 114 300 24 14 $C.Coral $true | Out-Null
    $months = @('SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY')
    for ($i = 0; $i -lt $months.Count; $i++) {
        $x = 64 + 94 * $i
        $fill = if ($i -lt 3) { $C.Teal } else { $C.Coral }
        $circle = $s.Shapes.AddShape($msoShapeOval, $x, 177, 38, 38)
        $circle.Fill.ForeColor.RGB = $fill
        $circle.Line.Visible = $msoFalse
        Add-Text $s $months[$i] ($x - 8) 225 54 24 11 $C.Gray $true $ppAlignCenter | Out-Null
        if ($i -lt $months.Count - 1) {
            $link = $s.Shapes.AddLine($x + 38, 196, $x + 94, 196)
            $link.Line.ForeColor.RGB = $C.LightGray
            $link.Line.Weight = 3
        }
    }
    Add-Text $s 'For every validation month' 58 294 270 30 20 $C.Navy $true | Out-Null
    foreach ($step in @(
        @{ N = '1'; X = 58; Text = 'Train on all permitted earlier sales' },
        @{ N = '2'; X = 350; Text = 'Rebuild historical encodings' },
        @{ N = '3'; X = 642; Text = 'Predict the next month' }
    )) {
        $badge = $s.Shapes.AddShape($msoShapeOval, $step.X, 343, 34, 34)
        $badge.Fill.ForeColor.RGB = $C.Teal
        $badge.Line.Visible = $msoFalse
        Add-Text $s $step.N ($step.X + 1) 347 32 24 13 $C.White $true $ppAlignCenter | Out-Null
        Add-Text $s $step.Text ($step.X + 48) 337 220 62 16 $C.Ink $false | Out-Null
    }
    Add-Text $s '10,069 homes per track' 58 442 320 34 22 $C.Teal $true | Out-Null
    Add-Text $s 'June to September 2026 remains the truly unseen hidden test' 414 446 478 30 16 $C.Gray $false $ppAlignRight | Out-Null
    Add-Notes $s @'
We used walk-forward validation instead of a random split. For each month from December 2025 through May 2026, every model trained only on sales from before that month and then predicted that month. Historical encodings followed the same rule within training, and homes sold on the same date could not reveal their targets to one another. We chose blend weights and calibration using September through November 2025. We did inspect model variants during development, so the hidden June through September test remains the truly unseen evaluation.

Sources: gbt_ensemble_experiments/results/final_walk_forward_report.json and report/report.tex.
'@

    # Slide 6
    $s = $presentation.Slides.Add(6, $ppLayoutBlank)
    $s.Background.Fill.ForeColor.RGB = $C.Cream
    Add-Title $s 'Validation results' 6
    Add-NativeChart $s $xlColumnClustered 52 112 560 344 @('Off-market', 'List-aware') @(
        @{ Name = 'Starter'; Values = @(0.208, 0.073); Color = $C.Gray },
        @{ Name = 'Final ensemble'; Values = @(0.0727, 0.0398); Color = $C.Teal }
    ) 0.25 | Out-Null
    Add-Text $s '7.27%' 662 133 210 48 34 $C.Teal $true $ppAlignCenter | Out-Null
    Add-Text $s 'off-market MdAPE' 662 181 210 25 15 $C.Gray $false $ppAlignCenter | Out-Null
    Add-Text $s '9.81% MAPE' 662 216 210 24 14 $C.Ink $true $ppAlignCenter | Out-Null
    Add-Text $s '3.98%' 662 282 210 48 34 $C.Coral $true $ppAlignCenter | Out-Null
    Add-Text $s 'list-aware MdAPE' 662 330 210 25 15 $C.Gray $false $ppAlignCenter | Out-Null
    Add-Text $s '5.19% MAPE' 662 365 210 24 14 $C.Ink $true $ppAlignCenter | Out-Null
    Add-Text $s 'MdAPE measures the typical percentage miss. Lower is better.' 64 476 820 28 14 $C.Gray $false | Out-Null
    Add-Notes $s @'
Across 10,069 validation homes per track, the final off-market model reached 7.27 percent MdAPE and 9.81 percent MAPE. The list-aware model reached 3.98 percent MdAPE and 5.19 percent MAPE. For context, the starter methods scored 20.8 percent off-market and 7.3 percent list-aware MdAPE. The improvement is largest off-market, which is also the harder track and counts twice in the competition.

Sources: gbt_ensemble_experiments/results/final_walk_forward_report.json and FINAL/handout.md.
'@

    # Slide 7
    $s = $presentation.Slides.Add(7, $ppLayoutBlank)
    $s.Background.Fill.ForeColor.RGB = $C.White
    Add-Title $s 'Where the model struggles' 7
    Add-NativeChart $s $xlBarClustered 42 108 612 354 @(
        'Highest price decile',
        'Semi-detached',
        'May 2026',
        '200 to 999 earlier sales',
        'At least 1,000 earlier sales'
    ) @(
        @{ Name = 'Off-market'; Values = @(0.1342, 0.0956, 0.0763, 0.0774, 0.0696); Color = $C.Teal },
        @{ Name = 'List-aware'; Values = @(0.0532, 0.0584, 0.0427, 0.0420, 0.0381); Color = $C.Coral }
    ) 0.15 | Out-Null
    Add-Text $s 'The risk pattern' 694 124 205 26 16 $C.Navy $true | Out-Null
    Add-Text $s "Expensive and semi-detached homes vary more.`n`nLess local history weakens the comparable-sales signal.`n`nMay's increase suggests market drift." 694 163 205 178 16 $C.Ink $false | Out-Null
    Add-Box $s 684 370 222 106 $C.LightTeal $C.LightTeal 1 | Out-Null
    Add-Text $s 'Responsible use' 700 386 190 24 15 $C.Teal $true | Out-Null
    Add-Text $s 'Flag unusual homes and show wider uncertainty ranges.' 700 419 190 46 16 $C.Ink $true | Out-Null
    Add-Text $s 'Segment differences are associations, not causal conclusions.' 54 488 600 20 12 $C.Gray $false | Out-Null
    Add-Notes $s @'
The errors are not evenly distributed. The highest price decile has 13.42 percent off-market MdAPE, almost twice the overall result. Semi-detached homes are also harder. Communities with fewer earlier sales perform worse than those with at least one thousand, which is consistent with weaker local evidence. May is worse than the earlier validation months, suggesting sensitivity to market drift. These patterns are associations, not causal claims. A responsible use of this model would flag expensive, rare, or low-history homes and attach a wider uncertainty range instead of presenting every estimate with the same confidence.

Source: video_prep/FAILURE_ANALYSIS.md, derived from gbt_ensemble_experiments/results/final_walk_forward_predictions.csv.
'@

    $presentation.SaveAs($OutputPath, $ppSaveAsOpenXMLPresentation)
    $presentation.Export($previewDir, 'PNG', 1920, 1080)
}
finally {
    if ($presentation) { $presentation.Close() }
    if ($ppt) { $ppt.Quit() }
    if ($script:excel) { $script:excel.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($script:excel) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output $OutputPath
