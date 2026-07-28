# Test Staff List Endpoint for St Mercy Hospital

Write-Host "🏥 TESTING STAFF LIST ENDPOINT FOR ST MERCY HOSPITAL" -ForegroundColor Cyan
Write-Host ("=" * 80)

# Step 1: Login
Write-Host "`n📋 STEP 1: Login as St Mercy Hospital Admin" -ForegroundColor Yellow
Write-Host ("-" * 80)
Write-Host "Email: alice.johnson@stmercy.hospital"
Write-Host "Password: password123"
Write-Host ""

$loginBody = @{
    email = "alice.johnson@stmercy.hospital"
    password = "password123"
} | ConvertTo-Json

try {
    $loginRes = Invoke-WebRequest -Uri "http://localhost:5001/api/auth/login" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $loginBody `
        -UseBasicParsing
    
    $loginData = $loginRes.Content | ConvertFrom-Json
    
    Write-Host "Response (truncated): " -ForegroundColor Green
    $loginData | Select-Object success, @{n="token";e={$_.token.Substring(0, 30) + "..."}} | ConvertTo-Json | Write-Host
    
    if (-not $loginData.token) {
        Write-Host "`n❌ LOGIN FAILED!" -ForegroundColor Red
        exit 1
    }
    
    $token = $loginData.token
    Write-Host "`n✅ LOGIN SUCCESSFUL!" -ForegroundColor Green
    Write-Host "User: $($loginData.user.name)" -ForegroundColor Green
    Write-Host "Hospital: $($loginData.user.hospitalName)" -ForegroundColor Green
    Write-Host "Role: $($loginData.user.role)" -ForegroundColor Green
    Write-Host "Hospital ID: $($loginData.user.hospitalId)" -ForegroundColor Green
    
    # Step 2: Get staff list
    Write-Host "`n📋 STEP 2: Query Staff List Endpoint" -ForegroundColor Yellow
    Write-Host ("-" * 80)
    Write-Host "URL: GET /api/manager/staff/list"
    Write-Host ""
    
    $staffRes = Invoke-WebRequest -Uri "http://localhost:5001/api/manager/staff/list" `
        -Method GET `
        -Headers @{"Authorization" = "Bearer $token"} `
        -UseBasicParsing
    
    $staffData = $staffRes.Content | ConvertFrom-Json
    
    Write-Host "Response Status: $($staffRes.StatusCode)" -ForegroundColor Green
    Write-Host "Response Body: " -ForegroundColor Green
    $staffData | ConvertTo-Json -Depth 2 | Write-Host
    
    if (-not $staffData.success) {
        Write-Host "`n❌ STAFF QUERY FAILED!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`n✅ STAFF QUERY SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Total Staff Found: $($staffData.count)" -ForegroundColor $(if ($staffData.count -gt 0) { "Green" } else { "Yellow" })
    
    if ($staffData.staff -and $staffData.staff.Count -gt 0) {
        Write-Host "`n👨‍⚕️  STAFF MEMBERS AT ST MERCY HOSPITAL:" -ForegroundColor Green
        $staffData.staff | ForEach-Object -Begin { $idx = 0 } -Process {
            $idx++
            Write-Host "`n$idx. $($_.firstName) $($_.lastName)" -ForegroundColor Green
            Write-Host "   Email: $($_.email)"
            Write-Host "   Role: $($_.systemRole)"
            $dept = if ($_.assignment.department) { $_.assignment.department } elseif ($_.department) { $_.department } else { 'N/A' }
            Write-Host "   Department: $dept"
            $status = if ($_.status) { $_.status } else { 'N/A' }
            Write-Host "   Status: $status"
        }
    } else {
        Write-Host "`n⚠️  NO STAFF MEMBERS FOUND!" -ForegroundColor Yellow
    }
    
    # Step 3: Test filters
    Write-Host "`n📋 STEP 3: Test Filters" -ForegroundColor Yellow
    Write-Host ("-" * 80)
    
    $filters = @(
        @{desc = "NURSES"; url = "/api/manager/staff/list?role=NURSE"},
        @{desc = "PHYSICIANS"; url = "/api/manager/staff/list?role=PHYSICIAN"},
        @{desc = 'Search: "Wong"'; url = "/api/manager/staff/list?search=Wong"}
    )
    
    foreach ($filter in $filters) {
        $filterRes = Invoke-WebRequest -Uri "http://localhost:5001$($filter.url)" `
            -Method GET `
            -Headers @{"Authorization" = "Bearer $token"} `
            -UseBasicParsing
        
        $filterData = $filterRes.Content | ConvertFrom-Json
        
        Write-Host "`n$($filter.desc):" -ForegroundColor Cyan
        Write-Host "  Count: $($filterData.count)"
        
        if ($filterData.staff -and $filterData.staff.Count -gt 0) {
            $filterData.staff | ForEach-Object {
                Write-Host "    - $($_.firstName) $($_.lastName) ($($_.systemRole))"
            }
        } else {
            Write-Host "    (No results)"
        }
    }
    
    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-Host "✅ TEST COMPLETE`n" -ForegroundColor Green
    
} catch {
    Write-Host "`n❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.InnerException) {
        Write-Host "Detail: $($_.Exception.InnerException.Message)" -ForegroundColor Red
    }
    exit 1
}
