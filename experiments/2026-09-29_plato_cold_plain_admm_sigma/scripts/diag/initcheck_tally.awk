# Complete tally of compute-sanitizer reports (run with --print-limit 0), streamed: one key per distinct
# (kernel, chain of named host frames); prints counts, the total, and how many reports have a cuSOLVER (cusolverDn*)
# host frame in their launch stack and how many do not. Report blocks start with "========= <kind> ..." and end with a
# bare "=========" line.
function flush() {
    if (inrep) {
        key = kind "\t" kern "\t" frames
        cnt[key]++
        if (frames ~ /cusolverDn/) insolver++; else outside++
        inrep = 0
    }
}
/^========= (Uninitialized|Invalid|Race|Barrier|Warp|Program hit|Unused|Leaked)/ {
    flush(); inrep = 1; total++; kind = $0; sub(/^========= /, "", kind); sub(/ of size [0-9]+ bytes/, "", kind)
    kern = ""; frames = ""; next
}
inrep && /^=========     at / {
    k = $0; sub(/^=========     at /, "", k); sub(/^void /, "", k); sub(/[(<].*/, "", k); sub(/\+0x[0-9a-f]+$/, "", k); kern = k; next
}
inrep && /Host Frame:/ {
    f = $0; sub(/.*Host Frame: */, "", f)
    if (f ~ /^\[0x/) next
    sub(/ \[0x[0-9a-f]+\].*/, "", f); sub(/\(.*/, "", f); sub(/<.*/, "", f); sub(/^void /, "", f)
    frames = frames ">" f; next
}
/^========= *$/ { flush(); next }
/ERROR SUMMARY:/ { summary = summary $0 "\n" }
END {
    flush()
    for (k in cnt) print cnt[k] "\t" k
    print "TOTAL_REPORTS\t" total + 0
    print "WITH_cusolverDn_FRAME\t" insolver + 0
    print "WITHOUT_cusolverDn_FRAME\t" outside + 0
    printf "%s", summary
}
