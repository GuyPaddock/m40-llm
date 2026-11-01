#!/usr/bin/env perl
# Preserve integer nanoseconds before analysis in double-precision JSON tools.
use strict;
use warnings;
use Config;
use JSON::PP;

die "64-bit Perl integers required\n" unless $Config{ivsize} >= 8;
my $json = JSON::PP->new->canonical;
my $records = 0;
my $origin;
while (my $line = <>) {
    die "activity input limit exceeded\n" if ++$records > 300004 || length($line) > 65536;
    my $record = $json->decode($line);
    if (($record->{kind} // '') eq 'cuda_kernel_activity') {
        my ($start, $end) = @{$record}{qw(start_ns end_ns)};
        die "invalid integer timestamps\n"
            unless defined($start) && defined($end) && $start =~ /^\d+$/ && $end =~ /^\d+$/
            && length($start) <= 19 && length($end) <= 19 && $start > 0 && $end > $start
            && $end <= 9223372036854775807;
        $origin //= $start;
        $record->{duration_ns} = $end - $start;
        $record->{relative_start_ns} = $start - $origin;
        $record->{relative_end_ns} = $end - $origin;
        # String absolute clocks so jq cannot round before subtracting them.
        $record->{start_ns} = "$start";
        $record->{end_ns} = "$end";
    }
    print $json->encode($record), "\n" or die "cannot write normalized activity: $!\n";
}
